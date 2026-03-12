"""
Training script for SurgicalPhaseLLM.

Usage:
    python train.py                        # use configs.yaml
    python train.py model.llm_model_name=gpt2 train.lr=5e-5

Video-by-video training loop:
  - One video at a time, clips of seq_len frames processed sequentially
  - Prompt KV cache built once per video and reused across clips
  - Context (detached hidden states) carried between clips, reset per video
  - WandB tracking for loss, accuracy, per-phase accuracy
"""

import os
import sys
import math
import random
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Make imports work from project root
sys.path.insert(0, os.path.dirname(__file__))

from models import SurgicalPhaseLLM, CHOLEC80_PHASES
from data.dataset import VideoClipDataset
from attn_analysis import get_region_groups


# ── Helpers ──────────────────────────────────────────────────────────────────

def masked_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Cross-entropy loss with label smoothing over valid frames.

    Args:
        logits:          (B, T, num_phases)
        labels:          (B, T)  int64
        mask:            (B, T)  bool, True = valid
        label_smoothing: smoothing factor (0 = standard CE)
    """
    B, T, C = logits.shape
    logits_flat = logits.view(B * T, C)
    labels_flat = labels.view(B * T)
    mask_flat   = mask.view(B * T)

    loss = nn.functional.cross_entropy(
        logits_flat, labels_flat,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    return loss[mask_flat].mean()



def temporal_smoothness_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Transition-aware temporal smoothness loss.

    Penalises abrupt distribution changes between adjacent frames, but
    automatically down-weights pairs near phase transitions: frames where
    the model is uncertain (high entropy) are likely transition frames, and
    forcing smoothness there would suppress legitimate sharp changes.

    Weight per adjacent pair (t, t+1):
        confidence_t  = 1 - H(p_t)  / H_max      ∈ [0, 1]
        pair_weight   = confidence_t * confidence_{t+1}
        L_smooth = mean( pair_weight * KL(p_t || p_{t+1}) )

    At a confident within-phase pair:  pair_weight ≈ 1 → full smoothness.
    At a transition boundary:          pair_weight ≈ 0 → loss suppressed.

    Args:
        logits: (B, T, num_phases)
        mask:   (B, T) bool, True = valid
    Returns:
        scalar loss
    """
    probs     = torch.softmax(logits, dim=-1)           # (B, T, C)
    log_probs = torch.log_softmax(logits, dim=-1)       # (B, T, C)

    # Adjacent-pair mask: both t and t+1 must be valid
    pair_mask = mask[:, :-1] & mask[:, 1:]              # (B, T-1)

    # KL(p_t || p_{t+1}) = sum p_t * (log p_t - log p_{t+1})
    kl = (probs[:, :-1] * (log_probs[:, :-1] - log_probs[:, 1:])).sum(-1)  # (B, T-1)

    if pair_mask.sum() == 0:
        return logits.new_tensor(0.0)

    # Entropy-based confidence weight: suppress loss at high-entropy (transition) frames
    C = logits.shape[-1]
    entropy = -(probs * torch.log(probs + 1e-8)).sum(-1)    # (B, T)
    confidence = 1.0 - entropy / math.log(C)                # (B, T) ∈ [0, 1]
    pair_conf = confidence[:, :-1] * confidence[:, 1:]      # (B, T-1)

    weighted_kl = pair_conf * kl                             # (B, T-1)
    return weighted_kl[pair_mask].mean()


def tool_aux_loss(
    tool_logits: torch.Tensor,
    tool_annots: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Auxiliary multi-label BCE loss for per-frame tool presence.

    Args:
        tool_logits: (B, T, 7)  raw logits
        tool_annots: (B, T, 7)  binary ground truth
        mask:        (B, T)     bool, True = valid frame
    """
    B, T, _ = tool_logits.shape
    mask_exp = mask.unsqueeze(-1).expand_as(tool_logits)    # (B, T, 7)
    loss = nn.functional.binary_cross_entropy_with_logits(
        tool_logits[mask_exp], tool_annots.float()[mask_exp], reduction="mean"
    )
    return loss


def hint_diversity_loss(hints: torch.Tensor) -> torch.Tensor:
    """
    Penalise redundancy among hint tokens by maximising pairwise dissimilarity.

    For each pair (i, j) of hint slots:
        L_div = mean max(0, cos_sim(h_i, h_j))   i ≠ j

    Pushes hints to specialise on different temporal segments / signals
    rather than all collapsing to the same dominant scene representation.

    Args:
        hints: (B, N_hints, d_llm)
    Returns:
        scalar loss
    """
    # Normalise to unit sphere
    h = nn.functional.normalize(hints, dim=-1)          # (B, N, d)
    # Pairwise cosine similarity matrix
    sim = torch.bmm(h, h.transpose(1, 2))               # (B, N, N)
    # Mask diagonal (self-similarity = 1, exclude)
    N = hints.shape[1]
    mask = ~torch.eye(N, dtype=torch.bool, device=hints.device)
    # Only penalise positive similarity (ReLU: don't reward orthogonality beyond 0)
    off_diag = sim[:, mask].view(hints.shape[0], N, N - 1)
    return torch.relu(off_diag).mean()


@torch.no_grad()
def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    """Returns (correct, total) counts over valid positions."""
    preds = logits.argmax(dim=-1)          # (B, T)
    valid_preds  = preds[mask]
    valid_labels = labels[mask]
    correct = (valid_preds == valid_labels).sum().item()
    total   = valid_labels.numel()
    return correct, total


# ── Test-time inference utilities ─────────────────────────────────────────────

@torch.no_grad()
def run_video_inference(
    model: SurgicalPhaseLLM,
    video_id: int,
    cfg,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collect per-frame logits and labels for an entire video (no grad).

    Processes clips sequentially (carrying memory/prev_visual like training),
    then concatenates only the *valid* frames across all clips.

    Returns:
        all_logits: (N_valid, num_phases)  — raw logits on CPU
        all_labels: (N_valid,)             — ground-truth labels on CPU
    """
    dataset = VideoClipDataset(
        video_id   = video_id,
        data_root  = cfg.data.data_root,
        phase_dir  = cfg.data.phase_annotation_dir,
        tool_dir   = cfg.data.tool_annotation_dir,
        seq_len    = cfg.data.seq_len,
        img_size   = cfg.data.img_size,
    )
    loader = DataLoader(
        dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = cfg.train.num_workers,
        pin_memory  = True,
    )

    prompt_kv   = model.build_prompt_kv()
    memory      = None
    prev_visual = None

    all_logits = []
    all_labels = []

    for clip_idx, (frames, tools, labels, valid_mask, frame_start) in enumerate(loader):
        frames     = frames.to(device)
        tools      = tools.to(device)
        labels     = labels.to(device)
        valid_mask = valid_mask.to(device)

        logits, _, memory, prev_visual, _, _ = model.forward_clip(
            frames      = frames,
            tool_annots = tools,
            memory      = memory,
            prev_visual = prev_visual,
            prompt_kv   = prompt_kv,
        )
        memory = memory.detach() if memory is not None else None

        # Collect only valid frames (B=1, so squeeze)
        mask_sq  = valid_mask[0]              # (T,)
        logits_v = logits[0][mask_sq]         # (N_valid, num_phases)
        labels_v = labels[0][mask_sq]         # (N_valid,)
        all_logits.append(logits_v.cpu())
        all_labels.append(labels_v.cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)




def compute_cls_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_phases: int,
) -> dict:
    """
    Compute frame-level acc, macro precision, recall, Jaccard.

    All metrics are macro-averaged across phases (equal weight per phase,
    regardless of class frequency) — standard for surgical phase evaluation.

    Returns dict with keys: acc, precision, recall, jaccard,
                            per_phase_precision, per_phase_recall, per_phase_jaccard
    """
    acc = (preds == labels).float().mean().item()

    precisions, recalls, jaccards = [], [], []
    for c in range(num_phases):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()

        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        recalls.append(   tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        jaccards.append(  tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0)

    return {
        "acc":                acc,
        "precision":          sum(precisions) / num_phases,
        "recall":             sum(recalls)    / num_phases,
        "jaccard":            sum(jaccards)   / num_phases,
        "per_phase_precision": precisions,
        "per_phase_recall":    recalls,
        "per_phase_jaccard":   jaccards,
    }


_ATTN_BAR_COLOURS = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#64b5cd", "#e07b39"]


@torch.no_grad()
def _collect_region_attn(model, video_id, cfg, device, max_clips=None):
    """
    Run a single video with output_attentions=True and accumulate per-region
    attention weights averaged over all layers, heads, and frames.

    Returns (avg_by_region dict, region_groups list), or (None, None) if empty.
    """
    dataset = VideoClipDataset(
        video_id  = video_id,
        data_root = cfg.data.data_root,
        phase_dir = cfg.data.phase_annotation_dir,
        tool_dir  = cfg.data.tool_annotation_dir,
        seq_len   = cfg.data.seq_len,
        img_size  = cfg.data.img_size,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    region_groups = get_region_groups(model.use_prev_clip)
    num_layers    = model.llm.config.num_hidden_layers
    prompt_kv     = model.build_prompt_kv()
    memory        = None
    prev_visual   = None

    region_sums = {li: {g: 0.0 for g, _ in region_groups} for li in range(num_layers)}
    n_clips = 0

    for i, (frames, tools, labels, valid_mask, frame_start) in enumerate(loader):
        if max_clips is not None and i >= max_clips:
            break
        frames = frames.to(device)
        tools  = tools.to(device)

        out = model.forward_clip(
            frames=frames, tool_annots=tools,
            memory=memory, prev_visual=prev_visual,
            prompt_kv=prompt_kv,
            output_attentions=True,
        )
        _, _, memory, prev_visual, _, _, attentions, regions = out
        memory = memory.detach() if memory is not None else None

        T_actual  = frames.shape[1]
        total_len = attentions[0].shape[-1]

        prompt_len_offset = regions.get("prompt", (0, 0))[1]
        for li in range(num_layers):
            a = attentions[li][0].float().cpu().numpy()   # (H, seq, seq)
            H = a.shape[0]
            # KV-cache mode:      a.shape[-2] == llm_input_len  (<  total_len)
            #                     → subtract prompt_len to convert shifted positions to row indices
            # Full-sequence mode: a.shape[-2] == total_len
            #                     → shifted positions are already correct row indices
            row_offset = 0 if a.shape[-2] == total_len else prompt_len_offset
            # Extract visual token rows
            s, e = regions["visual"]
            visual_rows = a[:, s - row_offset : e - row_offset, :]  # (H, T, total_len)
            for group_name, keys in region_groups:
                col_sum = 0.0
                for k in keys:
                    if k not in regions:
                        continue
                    s, e = regions[k]
                    e = min(e, total_len)
                    if s < total_len:
                        col_sum += float(visual_rows[:, :, s:e].sum())
                region_sums[li][group_name] += col_sum / (H * T_actual)
        n_clips += 1

    if n_clips == 0:
        return None, None

    group_names = [g for g, _ in region_groups]
    avg = {
        g: sum(region_sums[li][g] for li in range(num_layers)) / (num_layers * n_clips)
        for g in group_names
    }
    return avg, region_groups


def _make_attn_bar_figure(avg_by_region: dict, region_groups: list) -> plt.Figure:
    """Return a matplotlib Figure: bar chart of visual-token attention by region."""
    group_names = [g for g, _ in region_groups if avg_by_region.get(g, 0) > 0]
    values      = [avg_by_region[g] for g in group_names]

    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.bar(group_names, values,
                  color=_ATTN_BAR_COLOURS[:len(group_names)],
                  edgecolor="k", linewidth=0.5)
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Visual token → region (all layers avg)")
    if values:
        ax.set_ylim(0, max(values) * 1.25)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    return fig


@torch.no_grad()
def evaluate_test(model: SurgicalPhaseLLM, cfg, device: torch.device, epoch: int):
    """
    Reports: acc, macro precision, recall, Jaccard — printed and logged to WandB.
    """

    num_phases    = cfg.data.num_phases

    # All metrics computed per-video then averaged (equal weight per video)
    video_metrics = []

    model.eval()
    for video_id in cfg.data.test_videos:
        logits, labels = run_video_inference(model, video_id, cfg, device)
        preds    = logits.argmax(dim=-1)                                # (N,)

        video_metrics.append(compute_cls_metrics(preds, labels, num_phases))

    # Average each metric across videos
    m = {
        "acc":       sum(v["acc"]       for v in video_metrics) / len(video_metrics),
        "precision": sum(v["precision"] for v in video_metrics) / len(video_metrics),
        "recall":    sum(v["recall"]    for v in video_metrics) / len(video_metrics),
        "jaccard":   sum(v["jaccard"]   for v in video_metrics) / len(video_metrics),
    }

    print(
        f"\n{'='*60}\n"
        f"[TEST @ Epoch {epoch}] \n"
        f"  Acc={m['acc']:.4f}  Precision={m['precision']:.4f}"
        f"  Recall={m['recall']:.4f}  Jaccard={m['jaccard']:.4f}\n"
        f"{'='*60}\n"
    )

    wandb.log({
        "test/acc":       m["acc"],
        "test/precision": m["precision"],
        "test/recall":    m["recall"],
        "test/jaccard":   m["jaccard"],
        "epoch":          epoch,
    })

    # ── Attention bar chart visualization ─────────────────────────────────────
    max_clips = cfg.train.get("attn_log_max_clips", 20)
    if max_clips > 0 and wandb.run is not None:
        attn_video = cfg.data.test_videos[0]
        print(f"  [attn] Collecting attention weights (video {attn_video:02d}, "
              f"first {max_clips} clips)...")
        model.llm.set_attn_implementation("eager")
        try:
            avg_attn, region_groups = _collect_region_attn(
                model, attn_video, cfg, device, max_clips=max_clips,
            )
        finally:
            model.llm.set_attn_implementation("sdpa")
        if avg_attn is not None:
            fig = _make_attn_bar_figure(avg_attn, region_groups)
            wandb.log({"test/attn_bar": wandb.Image(fig), "epoch": epoch})
            plt.close(fig)
            print(f"  [attn] Logged attention bar chart to wandb.")

    return m


def build_optimizer(model: SurgicalPhaseLLM, cfg):
    lr = cfg.train.lr
    wd = cfg.train.weight_decay
    lora_factor = cfg.train.get("lora_lr_factor", 1.0)

    use_lora = cfg.model.get("use_lora", False)
    if use_lora and lora_factor != 1.0:
        lora_params  = [p for n, p in model.named_parameters()
                        if p.requires_grad and ("lora_A" in n or "lora_B" in n)]
        other_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and ("lora_A" not in n and "lora_B" not in n)]
        param_groups = [
            {"params": other_params, "lr": lr},
            {"params": lora_params,  "lr": lr * lora_factor},
        ]
        print(f"[optimizer] LoRA lr={lr * lora_factor:.2e} ({lora_factor}×), "
              f"other lr={lr:.2e}  "
              f"| LoRA params={sum(p.numel() for p in lora_params):,}, "
              f"other params={sum(p.numel() for p in other_params):,}")
    else:
        param_groups = [p for p in model.parameters() if p.requires_grad]

    if cfg.train.optimizer.lower() == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {cfg.train.optimizer}")


def build_scheduler(optimizer, cfg, total_steps: int):
    warmup_steps = cfg.train.warmup_epochs  # treated as warmup steps in per-video training
    if cfg.train.scheduler == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr_min_ratio = cfg.train.lr_min / cfg.train.lr
            return lr_min_ratio + (1.0 - lr_min_ratio) * cosine
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return None


# ── Per-video training pass ───────────────────────────────────────────────────

def run_video(
    model: SurgicalPhaseLLM,
    video_id: int,
    cfg,
    device: torch.device,
    optimizer=None,
    scheduler=None,
    scaler: GradScaler = None,
    is_train: bool = True,
):
    """
    Process all clips of one video sequentially.

    Returns dict with keys: loss, correct, total, per_phase_correct, per_phase_total
    """
    dataset = VideoClipDataset(
        video_id   = video_id,
        data_root  = cfg.data.data_root,
        phase_dir  = cfg.data.phase_annotation_dir,
        tool_dir   = cfg.data.tool_annotation_dir,
        seq_len    = cfg.data.seq_len,
        img_size   = cfg.data.img_size,
        is_train   = is_train,
    )
    loader = DataLoader(
        dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = cfg.train.num_workers,
        pin_memory  = True,
    )

    num_phases = cfg.data.num_phases
    total_loss        = 0.0
    total_loss_ce     = 0.0
    total_loss_smooth = 0.0
    total_loss_div    = 0.0
    total_loss_attn   = 0.0
    total_loss_tool   = 0.0
    total_loss_frame  = 0.0
    total_correct = 0
    total_frames  = 0
    phase_correct = defaultdict(int)
    phase_total   = defaultdict(int)

    # When LLM layers are trainable, prompt_kv must be rebuilt each window
    # (weights change after each optimizer step, cached KV would be stale).
    # When LLM is fully frozen, build once per video for efficiency.
    llm_is_trainable = any(p.requires_grad for p in model.llm.parameters())
    prompt_kv_static = None if (is_train and llm_is_trainable) else model.build_prompt_kv()

    # TBPTT: K clips per window — gradient flows within window, detach at boundary.
    # tbptt_k=1 → same as original (detach every clip).
    tbptt_k = cfg.train.get("tbptt_k", 1) if is_train else 1

    memory      = None  # CrossClipMemory global state, reset per video
    prev_visual = None  # Previous clip's visual_tokens, reset per video

    # TBPTT window state
    window_loss  = None   # accumulated loss tensor (keeps graph alive)
    window_clips = 0      # clips accumulated in current window
    clips = list(loader)
    n_clips = len(clips)

    if is_train:
        # accum_steps: how many TBPTT windows to accumulate before an optimizer step.
        # tbptt_k=1 → step every 4 clips (too-frequent updates on tiny windows hurt stability).
        # tbptt_k>1 → step every window (each window already spans several clips).
        accum_steps = 4 if tbptt_k == 1 else 1
        accum_count = 0
        optimizer.zero_grad()
        prompt_kv = model.build_prompt_kv() if llm_is_trainable else prompt_kv_static

    for clip_idx, (frames, tools, labels, valid_mask, frame_start) in enumerate(clips):
        # frames:      (1, T, 3, H, W)
        # tools:       (1, T, 7)
        # labels:      (1, T)
        # valid_mask:  (1, T)
        frames     = frames.to(device)
        tools      = tools.to(device)
        labels     = labels.to(device)
        valid_mask = valid_mask.to(device)

        if not is_train:
            prompt_kv = prompt_kv_static

        with torch.set_grad_enabled(is_train):
            with autocast("cuda", enabled=(scaler is not None)):
                logits, tool_logits, frame_aux_logits, memory, prev_visual, hints, attn_loss = model.forward_clip(
                    frames      = frames,
                    tool_annots = tools,
                    memory      = memory,
                    prev_visual = prev_visual,
                    prompt_kv   = prompt_kv,
                )
                loss_ce     = masked_ce_loss(logits, labels, valid_mask,
                                             label_smoothing=cfg.train.label_smoothing)
                loss_smooth = temporal_smoothness_loss(logits, valid_mask)
                loss_div    = hint_diversity_loss(hints)
                loss_tool   = (tool_aux_loss(tool_logits, tools, valid_mask)
                               if model.use_tool else loss_ce.new_tensor(0.0))
                loss_frame  = masked_ce_loss(frame_aux_logits, labels, valid_mask,
                                             label_smoothing=cfg.train.label_smoothing)
                loss = (loss_ce
                        + cfg.train.w_smooth     * loss_smooth
                        + cfg.train.w_diversity  * loss_div
                        + cfg.train.w_attn_focus * attn_loss
                        + cfg.train.w_tool       * loss_tool
                        + cfg.train.w_frame      * loss_frame)

        if is_train:
            window_loss   = loss if window_loss is None else window_loss + loss
            window_clips += 1
            is_last_clip  = (clip_idx == n_clips - 1)
            is_window_end = (window_clips >= tbptt_k) or is_last_clip

            if is_window_end:
                # TBPTT: backward per window to let gradient flow through memory chain.
                avg_loss = window_loss / window_clips
                if scaler is not None:
                    scaler.scale(avg_loss).backward()
                else:
                    avg_loss.backward()

                # Detach memory at TBPTT boundary
                if memory is not None:
                    memory = memory.detach()

                window_loss  = None
                window_clips = 0
                accum_count += 1

                # Optimizer step: every accum_steps windows, or at the last clip.
                if accum_count % accum_steps == 0 or is_last_clip:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        if cfg.train.grad_clip > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if cfg.train.grad_clip > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                        optimizer.step()
                    optimizer.zero_grad()

        # Accumulate metrics (detach for safety)
        total_loss        += loss.item()
        total_loss_ce     += loss_ce.item()
        total_loss_smooth += loss_smooth.item()
        total_loss_div    += loss_div.item()
        total_loss_attn   += attn_loss.item()
        total_loss_tool   += loss_tool.item()
        total_loss_frame  += loss_frame.item()
        correct, total = compute_accuracy(logits, labels, valid_mask)
        total_correct += correct
        total_frames  += total

        # Per-phase accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)            # (1, T)
            for ph in range(num_phases):
                ph_mask = valid_mask & (labels == ph)
                phase_total[ph]   += ph_mask.sum().item()
                phase_correct[ph] += (preds[ph_mask] == ph).sum().item()

    # Scheduler steps once per video (LR decay paced by number of videos processed).
    if is_train and scheduler is not None:
        scheduler.step()

    num_clips = max(1, len(loader))
    return {
        "loss":        total_loss        / num_clips,
        "loss_ce":     total_loss_ce     / num_clips,
        "loss_smooth": total_loss_smooth / num_clips,
        "loss_div":    total_loss_div    / num_clips,
        "loss_attn":   total_loss_attn   / num_clips,
        "loss_tool":   total_loss_tool   / num_clips,
        "loss_frame":  total_loss_frame  / num_clips,
        "correct":       total_correct,
        "total":         total_frames,
        "phase_correct": phase_correct,
        "phase_total":   phase_total,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", help="OmegaConf dot-path overrides")
    args = parser.parse_args()

    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "configs.yaml"))
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    torch.manual_seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SurgicalPhaseLLM.from_config(cfg).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")
    # Per-module breakdown to verify freeze settings
    for name, module in model.named_children():
        t = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if t > 0:
            print(f"  [trainable] {name}: {t:,}")

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)
    total_steps = cfg.train.epochs * len(cfg.data.train_videos)
    scheduler = build_scheduler(optimizer, cfg, total_steps)

    # ── Mixed precision ───────────────────────────────────────────────────────
    use_amp = device.type == "cuda" and cfg.train.get("use_amp", True)
    scaler  = GradScaler("cuda") if use_amp else None
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb.init(
        project = "surgical-phase-llm",
        config  = OmegaConf.to_container(cfg, resolve=True),
        name    = f"{cfg.model.llm_model_name.split('/')[-1]}_1video",
    )
    wandb.watch(model, log="gradients", log_freq=100)

    os.makedirs(cfg.train.save_dir, exist_ok=True)
    best_val_acc = 0.0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()

        epoch_loss        = 0.0
        epoch_loss_ce     = 0.0
        epoch_loss_smooth = 0.0
        epoch_loss_div    = 0.0
        epoch_loss_attn   = 0.0
        epoch_loss_tool   = 0.0
        epoch_loss_frame  = 0.0
        epoch_correct = 0
        epoch_total   = 0
        epoch_phase_correct = defaultdict(int)
        epoch_phase_total   = defaultdict(int)

        train_videos = list(cfg.data.train_videos)
        random.shuffle(train_videos)

        for step, video_id in enumerate(train_videos, 1):
            result = run_video(
                model, video_id, cfg, device,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler, is_train=True,
            )
            epoch_loss        += result["loss"]
            epoch_loss_ce     += result["loss_ce"]
            epoch_loss_smooth += result["loss_smooth"]
            epoch_loss_div    += result["loss_div"]
            epoch_loss_attn   += result["loss_attn"]
            epoch_loss_tool   += result["loss_tool"]
            epoch_loss_frame  += result["loss_frame"]
            epoch_correct += result["correct"]
            epoch_total   += result["total"]
            for ph in range(cfg.data.num_phases):
                epoch_phase_correct[ph] += result["phase_correct"][ph]
                epoch_phase_total[ph]   += result["phase_total"][ph]

            # Step-level log
            if step % cfg.train.log_interval == 0 or step == len(cfg.data.train_videos):
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch} | video {video_id:02d}/{cfg.data.train_videos[-1]} "
                      f"| loss {result['loss']:.4f} | lr {lr_now:.2e}")
                wandb.log({
                    "train/video_loss": result["loss"],
                    "train/lr":         lr_now,
                    "epoch":            epoch,
                    "step":             (epoch - 1) * len(cfg.data.train_videos) + step,
                })

        # Epoch-level train metrics
        n_vid = len(cfg.data.train_videos)
        train_acc         = epoch_correct     / max(1, epoch_total)
        train_loss        = epoch_loss        / n_vid
        train_loss_ce     = epoch_loss_ce     / n_vid
        train_loss_smooth = epoch_loss_smooth / n_vid
        train_loss_div    = epoch_loss_div    / n_vid
        train_loss_attn   = epoch_loss_attn   / n_vid
        train_loss_tool   = epoch_loss_tool   / n_vid
        train_loss_frame  = epoch_loss_frame  / n_vid

        wandb.log({
            "train/loss":        train_loss,
            "train/loss_ce":     train_loss_ce,
            "train/loss_smooth": train_loss_smooth,
            "train/loss_div":    train_loss_div,
            "train/loss_attn":   train_loss_attn,
            "train/loss_tool":   train_loss_tool,
            "train/loss_frame":  train_loss_frame,
            "train/accuracy":    train_acc,
            "epoch":             epoch,
        })

        print(f"\n=== Epoch {epoch} Train | loss {train_loss:.4f} "
              f"(ce={train_loss_ce:.3f} sm={train_loss_smooth:.3f} "
              f"div={train_loss_div:.3f} attn={train_loss_attn:.3f} "
              f"tool={train_loss_tool:.3f} frame={train_loss_frame:.3f})"
              f" | acc {train_acc:.4f} ===")

        # ── Test evaluation every epoch ───────────────────────────────────────
        m = evaluate_test(model, cfg, device, epoch)
        model.train()   # restore train mode after evaluation

        # Save best checkpoint based on test accuracy
        if m["acc"] > best_val_acc:
            best_val_acc = m["acc"]
            ckpt_path = os.path.join(cfg.train.save_dir, "best.pt")
            torch.save({
                "epoch":    epoch,
                "model":    model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "test_acc": m["acc"],
                "cfg":      OmegaConf.to_container(cfg),
            }, ckpt_path)
            print(f"  Saved best checkpoint → {ckpt_path} (test_acc={m['acc']:.4f})")

        # Save latest checkpoint every epoch
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg":       OmegaConf.to_container(cfg),
        }, os.path.join(cfg.train.save_dir, "latest.pt"))

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()

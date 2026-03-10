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

    for clip_idx, (frames, tools, labels, valid_mask) in enumerate(loader):
        frames     = frames.to(device)
        tools      = tools.to(device)
        labels     = labels.to(device)
        valid_mask = valid_mask.to(device)

        logits, memory, prev_visual, _, _ = model.forward_clip(
            frames      = frames,
            tool_annots = tools,
            memory      = memory,
            prev_visual = prev_visual,
            prompt_kv   = prompt_kv,
            clip_idx    = clip_idx,
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


@torch.no_grad()
def evaluate_test(model: SurgicalPhaseLLM, cfg, device: torch.device, epoch: int):
    """
    Evaluate on test videos with accuracy-boosting post-processing:
      1. Temporal soft smoothing  (sliding window probability average)
      2. Short-segment removal    (merge isolated phase blips)

    Reports: acc, macro precision, recall, Jaccard — printed and logged to WandB.
    """
    smooth_window = cfg.train.get("test_smooth_window", 15)
    min_seg_len   = cfg.train.get("test_min_segment",  10)
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
        f"[TEST @ Epoch {epoch}]  (smooth={smooth_window}fr, min_seg={min_seg_len}fr)\n"
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
        optimizer.zero_grad()
        prompt_kv = model.build_prompt_kv() if llm_is_trainable else prompt_kv_static

    for clip_idx, (frames, tools, labels, valid_mask) in enumerate(clips):
        # frames:     (1, T, 3, H, W)
        # tools:      (1, T, 7)
        # labels:     (1, T)
        # valid_mask: (1, T)
        frames     = frames.to(device)
        tools      = tools.to(device)
        labels     = labels.to(device)
        valid_mask = valid_mask.to(device)

        if not is_train:
            prompt_kv = prompt_kv_static

        with torch.set_grad_enabled(is_train):
            with autocast("cuda", enabled=(scaler is not None)):
                logits, memory, prev_visual, hints, attn_loss = model.forward_clip(
                    frames      = frames,
                    tool_annots = tools,
                    memory      = memory,
                    prev_visual = prev_visual,
                    prompt_kv   = prompt_kv,
                    clip_idx    = clip_idx,
                )
                loss_ce     = masked_ce_loss(logits, labels, valid_mask,
                                             label_smoothing=cfg.train.label_smoothing)
                loss_smooth = temporal_smoothness_loss(logits, valid_mask)
                loss_div    = hint_diversity_loss(hints)
                loss = (loss_ce
                        + cfg.train.w_smooth     * loss_smooth
                        + cfg.train.w_diversity  * loss_div
                        + cfg.train.w_attn_focus * attn_loss)

        if is_train:
            window_loss   = loss if window_loss is None else window_loss + loss
            window_clips += 1
            is_last_clip  = (clip_idx == n_clips - 1)
            is_window_end = (window_clips >= tbptt_k) or is_last_clip

            if is_window_end:
                avg_loss = window_loss / window_clips
                if scaler is not None:
                    scaler.scale(avg_loss).backward()
                    scaler.unscale_(optimizer)
                    if cfg.train.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    avg_loss.backward()
                    if cfg.train.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                # Detach memory at window boundary so next window starts fresh
                if memory is not None:
                    memory = memory.detach()

                optimizer.zero_grad()
                # Rebuild prompt KV after optimizer step (LoRA weights updated)
                if llm_is_trainable and not is_last_clip:
                    prompt_kv = model.build_prompt_kv()

                window_loss  = None
                window_clips = 0

        # Accumulate metrics (detach for safety)
        total_loss        += loss.item()
        total_loss_ce     += loss_ce.item()
        total_loss_smooth += loss_smooth.item()
        total_loss_div    += loss_div.item()
        total_loss_attn   += attn_loss.item()
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

    num_clips = max(1, len(loader))
    return {
        "loss":        total_loss        / num_clips,
        "loss_ce":     total_loss_ce     / num_clips,
        "loss_smooth": total_loss_smooth / num_clips,
        "loss_div":    total_loss_div    / num_clips,
        "loss_attn":   total_loss_attn   / num_clips,
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

        wandb.log({
            "train/loss":        train_loss,
            "train/loss_ce":     train_loss_ce,
            "train/loss_smooth": train_loss_smooth,
            "train/loss_div":    train_loss_div,
            "train/loss_attn":   train_loss_attn,
            "train/accuracy":    train_acc,
            "epoch":             epoch,
        })

        print(f"\n=== Epoch {epoch} Train | loss {train_loss:.4f} "
              f"(ce={train_loss_ce:.3f} sm={train_loss_smooth:.3f} "
              f"div={train_loss_div:.3f} attn={train_loss_attn:.3f})"
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

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


def next_frame_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Auxiliary next-frame phase prediction loss.

    At position t, use logits[t] to predict label[t+1].
    Forces the model to anticipate upcoming phases — directly learns phase
    transition signals without any heuristic weighting.

    Only valid where both frame t and t+1 have valid labels.

    Args:
        logits:          (B, T, num_phases)
        labels:          (B, T)  int64
        mask:            (B, T)  bool, True = valid
        label_smoothing: smoothing factor
    """
    B, T, C = logits.shape

    # logits at t predicting labels at t+1
    logits_shifted = logits[:, :-1].reshape(-1, C)   # (B*(T-1), C)
    labels_shifted = labels[:, 1:].reshape(-1)        # (B*(T-1),)
    pair_mask      = (mask[:, :-1] & mask[:, 1:]).reshape(-1)  # (B*(T-1),)

    if pair_mask.sum() == 0:
        return logits.new_tensor(0.0)

    loss = nn.functional.cross_entropy(
        logits_shifted, labels_shifted,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    return loss[pair_mask].mean()


def temporal_smoothness_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Penalise abrupt changes in predicted phase distribution between all adjacent frames.

    Uses KL divergence between consecutive softmax outputs:
        L_smooth = mean KL(p_t || p_{t+1})  over all valid adjacent pairs

    Applied to ALL pairs (including phase-transition boundaries), so it:
      - suppresses oscillation within a phase (consistency)
      - encourages gradual distribution shift at transitions rather than sudden jumps
    Works in tandem with next_frame_ce_loss: next_frame_ce teaches WHEN to transition,
    smooth loss ensures the transition happens GRADUALLY.

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
    return kl[pair_mask].mean()


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

    for frames, tools, labels, valid_mask in loader:
        frames     = frames.to(device)
        tools      = tools.to(device)
        labels     = labels.to(device)
        valid_mask = valid_mask.to(device)

        logits, memory, prev_visual, _, _, _ = model.forward_clip(
            frames      = frames,
            tool_annots = tools,
            memory      = memory,
            prev_visual = prev_visual,
            prompt_kv   = prompt_kv,
        )

        # Collect only valid frames (B=1, so squeeze)
        mask_sq  = valid_mask[0]              # (T,)
        logits_v = logits[0][mask_sq]         # (N_valid, num_phases)
        labels_v = labels[0][mask_sq]         # (N_valid,)
        all_logits.append(logits_v.cpu())
        all_labels.append(labels_v.cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def temporal_smooth_logits(logits: torch.Tensor, window: int = 15) -> torch.Tensor:
    """
    Soft temporal smoothing: average softmax probabilities over a sliding window.

    Applies symmetric avg-pool over the time axis so each frame's final
    probability is the mean of its ±(window//2) neighbours.  Preserves
    temporal length exactly.

    Args:
        logits: (N, C) — raw per-frame logits
        window: sliding window size (odd recommended)
    Returns:
        (N, C) — smoothed probability vectors (sum to 1 along C)
    """
    probs = torch.softmax(logits, dim=-1)               # (N, C)
    # avg_pool1d expects (B, C, L) — treat frames as the length dimension
    probs_t  = probs.T.unsqueeze(0)                     # (1, C, N)
    pad      = window // 2
    smoothed = nn.functional.avg_pool1d(
        probs_t, kernel_size=window, stride=1, padding=pad
    )
    return smoothed.squeeze(0).T                        # (N, C)


def remove_short_segments(preds: torch.Tensor, min_len: int = 10) -> torch.Tensor:
    """
    Remove phase segments shorter than min_len frames.

    Short isolated segments (e.g., a 3-frame "blip" of phase 2 inside phase 1)
    are replaced by the left neighbour (or right neighbour at the video start).
    Iterates until no segment shorter than min_len remains.

    Args:
        preds:   (N,) int64 frame-level predictions
        min_len: minimum acceptable segment length in frames
    Returns:
        (N,) cleaned predictions
    """
    p = preds.tolist()
    N = len(p)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < N:
            ph = p[i]
            j  = i
            while j < N and p[j] == ph:
                j += 1
            if (j - i) < min_len:
                # Replace with left neighbour; fall back to right if at start
                fill = p[i - 1] if i > 0 else (p[j] if j < N else ph)
                for k in range(i, j):
                    p[k] = fill
                changed = True
            i = j
    return torch.tensor(p, dtype=preds.dtype)


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

    all_preds  = []
    all_labels = []

    model.eval()
    for video_id in cfg.data.test_videos:
        logits, labels = run_video_inference(model, video_id, cfg, device)

        # 1. Temporal soft smoothing
        smoothed = temporal_smooth_logits(logits, window=smooth_window)  # (N, C)
        preds    = smoothed.argmax(dim=-1)                                # (N,)

        # 2. Short-segment removal
        preds = remove_short_segments(preds, min_len=min_seg_len)

        all_preds.append(preds)
        all_labels.append(labels)

    all_preds  = torch.cat(all_preds,  dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    m = compute_cls_metrics(all_preds, all_labels, num_phases)

    # Per-phase Jaccard table
    phase_jaccard_str = "  ".join(
        f"{CHOLEC80_PHASES[i][:6]}={m['per_phase_jaccard'][i]:.3f}"
        for i in range(num_phases)
    )
    print(
        f"\n{'='*60}\n"
        f"[TEST @ Epoch {epoch}]  (smooth={smooth_window}fr, min_seg={min_seg_len}fr)\n"
        f"  Acc={m['acc']:.4f}  Precision={m['precision']:.4f}"
        f"  Recall={m['recall']:.4f}  Jaccard={m['jaccard']:.4f}\n"
        f"  Per-phase Jaccard: {phase_jaccard_str}\n"
        f"{'='*60}\n"
    )

    log_dict = {
        "test/acc":       m["acc"],
        "test/precision": m["precision"],
        "test/recall":    m["recall"],
        "test/jaccard":   m["jaccard"],
        "epoch":          epoch,
    }
    for i, ph in enumerate(CHOLEC80_PHASES):
        log_dict[f"test/jaccard_{ph}"]   = m["per_phase_jaccard"][i]
        log_dict[f"test/precision_{ph}"] = m["per_phase_precision"][i]
        log_dict[f"test/recall_{ph}"]    = m["per_phase_recall"][i]
    wandb.log(log_dict)

    return m


def build_optimizer(model: SurgicalPhaseLLM, cfg):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if cfg.train.optimizer.lower() == "adamw":
        return torch.optim.AdamW(trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
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
    )
    loader = DataLoader(
        dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = cfg.train.num_workers,
        pin_memory  = True,
    )

    num_phases = cfg.data.num_phases
    total_loss = 0.0
    total_correct = 0
    total_frames  = 0
    phase_correct = defaultdict(int)
    phase_total   = defaultdict(int)

    # Build prompt KV once per video (fixed prompt, reused across all clips)
    prompt_kv   = model.build_prompt_kv()
    memory      = None  # CrossClipMemory global state, reset per video
    prev_visual = None  # Previous clip's visual_tokens, reset per video

    for frames, tools, labels, valid_mask in loader:
        # frames:     (1, T, 3, H, W)
        # tools:      (1, T, 7)
        # labels:     (1, T)
        # valid_mask: (1, T)
        frames     = frames.to(device)
        tools      = tools.to(device)
        labels     = labels.to(device)
        valid_mask = valid_mask.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            with autocast("cuda", enabled=(scaler is not None)):
                logits, memory, prev_visual, hints, attn_loss, lm_phase_logits = model.forward_clip(
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
                # Auxiliary next-frame prediction via frozen lm_head:
                # at position t, lm_phase_logits[t] predicts label[t+1].
                # Forces reprogramming to produce representations that the
                # LLM's own next-token head decodes as the upcoming phase.
                loss_next   = next_frame_ce_loss(lm_phase_logits, labels, valid_mask,
                                                 label_smoothing=cfg.train.label_smoothing)
                loss = (loss_ce
                        + cfg.train.w_smooth      * loss_smooth
                        + cfg.train.w_diversity   * loss_div
                        + cfg.train.w_attn_focus  * attn_loss
                        + cfg.train.w_next_frame  * loss_next)

        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if cfg.train.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.train.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # Accumulate metrics
        total_loss += loss.item()
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

    num_clips = len(loader)
    return {
        "loss":          total_loss / max(1, num_clips),
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

        epoch_loss    = 0.0
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
            epoch_loss    += result["loss"]
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
        train_acc = epoch_correct / max(1, epoch_total)
        train_loss = epoch_loss / len(cfg.data.train_videos)
        phase_accs = {
            CHOLEC80_PHASES[ph]: epoch_phase_correct[ph] / max(1, epoch_phase_total[ph])
            for ph in range(cfg.data.num_phases)
        }

        log_dict = {
            "train/epoch_loss": train_loss,
            "train/accuracy":   train_acc,
            "epoch":            epoch,
        }
        log_dict.update({f"train/acc_{ph}": acc for ph, acc in phase_accs.items()})
        wandb.log(log_dict)

        print(f"\n=== Epoch {epoch} Train | loss {train_loss:.4f} | acc {train_acc:.4f} ===")

        # ── Validation ────────────────────────────────────────────────────────
        if epoch % cfg.train.eval_interval == 0:
            model.eval()
            val_loss    = 0.0
            val_correct = 0
            val_total   = 0
            val_phase_correct = defaultdict(int)
            val_phase_total   = defaultdict(int)

            for video_id in cfg.data.val_videos:
                result = run_video(model, video_id, cfg, device, is_train=False)
                val_loss    += result["loss"]
                val_correct += result["correct"]
                val_total   += result["total"]
                for ph in range(cfg.data.num_phases):
                    val_phase_correct[ph] += result["phase_correct"][ph]
                    val_phase_total[ph]   += result["phase_total"][ph]

            val_acc  = val_correct / max(1, val_total)
            val_loss = val_loss / len(cfg.data.val_videos)
            val_phase_accs = {
                CHOLEC80_PHASES[ph]: val_phase_correct[ph] / max(1, val_phase_total[ph])
                for ph in range(cfg.data.num_phases)
            }

            log_dict = {
                "val/loss":     val_loss,
                "val/accuracy": val_acc,
                "epoch":        epoch,
            }
            log_dict.update({f"val/acc_{ph}": acc for ph, acc in val_phase_accs.items()})
            wandb.log(log_dict)

            print(f"=== Epoch {epoch} Val   | loss {val_loss:.4f} | acc {val_acc:.4f} ===\n")

            # Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = os.path.join(cfg.train.save_dir, "best.pt")
                torch.save({
                    "epoch":      epoch,
                    "model":      model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "val_acc":    val_acc,
                    "cfg":        OmegaConf.to_container(cfg),
                }, ckpt_path)
                print(f"  Saved best checkpoint → {ckpt_path} (val_acc={val_acc:.4f})")

        # ── Test evaluation (every test_eval_interval epochs) ─────────────────
        test_interval = cfg.train.get("test_eval_interval", 10)
        if epoch % test_interval == 0:
            evaluate_test(model, cfg, device, epoch)
            model.train()   # restore train mode after evaluation

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

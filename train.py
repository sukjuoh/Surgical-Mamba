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
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Cross-entropy loss with label smoothing over valid (non-padded) positions.

    Label smoothing (α=0.1): softens hard targets to (1-α)*one_hot + α/C.
    Reduces overconfidence and improves calibration.

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


def temporal_smoothness_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Penalise abrupt changes in predicted phase distribution between adjacent frames.

    Uses KL divergence between consecutive softmax outputs:
        L_smooth = mean KL(p_t || p_{t+1})  over valid adjacent pairs

    Surgical phases last minutes — adjacent frames should have nearly identical
    distributions unless a genuine phase transition is occurring.

    Args:
        logits: (B, T, num_phases)
        mask:   (B, T) bool, True = valid
    Returns:
        scalar loss
    """
    probs = torch.softmax(logits, dim=-1)               # (B, T, C)
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
    prompt_kv = model.build_prompt_kv()
    context   = None

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
                logits, context, hints, attn_loss = model.forward_clip(
                    frames     = frames,
                    tool_annots= tools,
                    context    = context,
                    prompt_kv  = prompt_kv,
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

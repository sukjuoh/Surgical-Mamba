"""Training script for CausalSurgicalMamba.

Usage::

    python train.py
    python train.py model.d_model=512 train.lr=5e-4

Video-by-video training loop: one video at a time, clips of ``seq_len``
frames processed sequentially. The slow SSM state carries across clips and
is detached at the TBPTT boundary.
"""

import os
import sys
import math
import random
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from models import CausalSurgicalMamba
from models.causal_surgical_mamba import OnlineSession
from data.dataset import VideoClipDataset


# ── Dataset registry ──────────────────────────────────────────────────────────
# Standard public splits for each supported dataset.
#   train_videos / eval_videos: integer IDs passed to VideoClipDataset
#   tag_format:                 train-split directory tag
#   eval_tag_format:            val/test-split directory tag (may differ for
#                               datasets with separate train/test naming, e.g.
#                               M2CAI16's "workflow_video_*" vs "test_workflow_video_*")
DATASET_SPLITS = {
    "cholec80": {
        "train_videos":    list(range(1, 41)),
        "val_videos":      list(range(33, 41)),
        "test_videos":     list(range(41, 81)),
        "tag_format":      "video{:02d}",
        "eval_tag_format": "video{:02d}",
        "num_phases":      7,
    },
    "m2cai16": {
        "train_videos":    list(range(1, 28)),
        "val_videos":      list(range(21, 28)),
        "test_videos":     list(range(1, 15)),
        "tag_format":      "workflow_video_{:02d}",
        "eval_tag_format": "test_workflow_video_{:02d}",
        "num_phases":      8,
    },
    "autolaparo": {
        "train_videos":    list(range(1, 11)),
        "val_videos":      list(range(11, 15)),
        "test_videos":     list(range(15, 22)),
        "tag_format":      "{:02d}",
        "eval_tag_format": "{:02d}",
        "num_phases":      7,
    },
}


def _splits(cfg) -> dict:
    name = cfg.data.dataset
    if name not in DATASET_SPLITS:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {list(DATASET_SPLITS)}")
    return DATASET_SPLITS[name]


# ── Helpers ───────────────────────────────────────────────────────────────────

def masked_ce_loss(logits, labels, mask, label_smoothing=0.0):
    B, T, C = logits.shape
    loss = F.cross_entropy(
        logits.view(B * T, C), labels.view(B * T),
        label_smoothing=label_smoothing, reduction="none",
    )
    return loss[mask.view(B * T)].mean()


def _dataset_kwargs(cfg, is_train: bool = False, tag_format: str = None) -> dict:
    return dict(
        data_root=cfg.data.data_root,
        phase_dir=cfg.data.phase_annotation_dir,
        tool_dir=cfg.data.tool_annotation_dir,
        seq_len=cfg.data.seq_len, img_size=cfg.data.img_size,
        is_train=is_train,
        tag_format=tag_format or _splits(cfg)["tag_format"],
    )


def _tag_format(cfg, split: str) -> str:
    """Tag format per split. M2CAI16 uses a separate prefix for the test split;
    the val split reuses the train naming (it is a subset of train videos)."""
    s = _splits(cfg)
    return s["eval_tag_format"] if split == "test" else s["tag_format"]


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



def gaussian_transition_map(labels, sigma_l=2.0, sigma_r=12.0):
    """
    LoVIT asymmetric Gaussian transition map.
    labels: (B, T) long
    Returns: (B, T) float in [0, 1]
    """
    B, T = labels.shape
    device = labels.device
    t_idx = torch.arange(T, device=device, dtype=torch.float)
    out = torch.zeros(B, T, device=device)

    for b in range(B):
        bpts = (labels[b, 1:] != labels[b, :-1]).nonzero(as_tuple=False).view(-1)
        for bpt in bpts:
            bpt_f = float(bpt.item())
            g = torch.zeros(T, device=device)

            left  = (t_idx > bpt_f - 3 * sigma_l) & (t_idx < bpt_f)
            right = (t_idx > bpt_f)                & (t_idx < bpt_f + 3 * sigma_r)
            at    = t_idx == bpt_f

            g[left]  = torch.exp(-(t_idx[left]  - bpt_f) ** 2 / (2 * sigma_l ** 2))
            g[right] = torch.exp(-(t_idx[right] - bpt_f) ** 2 / (2 * sigma_r ** 2))
            g[at]    = 1.0

            out[b] = torch.maximum(out[b], g)

    return out


def transition_l1_loss(trans_logits, labels, mask):
    """L1 between raw trans_logits and gaussian transition map, masked."""
    target = gaussian_transition_map(labels)           # (B, T)
    pred   = trans_logits.squeeze(-1)                  # (B, T)
    loss   = (pred - target).abs()
    return loss[mask].mean()


@torch.no_grad()
def compute_accuracy(logits, labels, mask):
    preds = logits.argmax(dim=-1)
    correct = (preds[mask] == labels[mask]).sum().item()
    return correct, labels[mask].numel()


# ── Test-time inference & evaluation ──────────────────────────────────────────

@torch.no_grad()
def run_video_inference(model, video_id, cfg, device, tag_format: str = None):
    dataset = VideoClipDataset(
        video_id=video_id,
        **_dataset_kwargs(cfg, is_train=False, tag_format=tag_format),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=cfg.train.num_workers, pin_memory=True)

    slow_states = None
    all_logits, all_labels = [], []
    for frames, _, labels, valid_mask, _ in loader:
        frames     = frames.to(device)
        valid_mask = valid_mask.to(device)
        logits, slow_states, _, _ = model.forward_clip(frames, slow_states=slow_states)
        if slow_states is not None:
            slow_states = [
                tuple(s_i.detach() for s_i in s) if s is not None else None
                for s in slow_states
            ]
        mask_sq = valid_mask[0].cpu()
        all_logits.append(logits[0].cpu()[mask_sq])
        all_labels.append(labels[0][mask_sq])

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def compute_cls_metrics(preds, labels, num_phases):
    acc = (preds == labels).float().mean().item()
    precisions, recalls, jaccards = [], [], []
    for c in range(num_phases):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        jaccards.append(tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0)
    return {
        "acc":       acc,
        "precision": sum(precisions) / num_phases,
        "recall":    sum(recalls) / num_phases,
        "jaccard":   sum(jaccards) / num_phases,
    }


@torch.no_grad()
def _evaluate_videos(model, video_ids, cfg, device, epoch, tag: str):
    num_phases = cfg.data.num_phases
    model.eval()
    tag_format = _tag_format(cfg, tag)

    clip_metrics = []
    for video_id in video_ids:
        logits, labels = run_video_inference(
            model, video_id, cfg, device, tag_format=tag_format,
        )
        clip_metrics.append(compute_cls_metrics(logits.argmax(-1), labels, num_phases))

    def avg(metrics, key):
        return sum(v[key] for v in metrics) / len(metrics)

    mc = {k: avg(clip_metrics, k) for k in ("acc", "precision", "recall", "jaccard")}

    print(
        f"\n{'='*60}\n"
        f"[{tag.upper()} @ Epoch {epoch}]\n"
        f"  Clip — Acc={mc['acc']:.4f}  Prec={mc['precision']:.4f}"
        f"  Rec={mc['recall']:.4f}  Jac={mc['jaccard']:.4f}\n"
        f"{'='*60}\n"
    )
    wandb.log({
        f"{tag}_clip/acc":       mc["acc"],
        f"{tag}_clip/precision": mc["precision"],
        f"{tag}_clip/recall":    mc["recall"],
        f"{tag}_clip/jaccard":   mc["jaccard"],
        "epoch":                 epoch,
    })
    return mc


@torch.no_grad()
def evaluate_val(model, cfg, device, epoch):
    return _evaluate_videos(model, _splits(cfg)["val_videos"], cfg, device, epoch, tag="val")


@torch.no_grad()
def evaluate_test(model, cfg, device, epoch):
    return _evaluate_videos(model, _splits(cfg)["test_videos"], cfg, device, epoch, tag="test")


# ── Optimizer & scheduler ─────────────────────────────────────────────────────

def _convnext_depth(name: str, n_stages: int) -> int | None:
    """Return depth ∈ [0, n_stages] for a ConvNeXt backbone param name, or None."""
    if name.startswith("extractor.backbone.stem"):
        return 0
    for i in range(n_stages):
        if f"extractor.backbone.stages.{i}." in name:
            return i + 1
    if name.startswith("extractor.backbone.norm_pre"):
        return n_stages
    return None


def _split_wd(group, wd: float):
    """Split a param group (dict or list of params) into [decay, no_decay] subgroups.

    Honors the `_no_weight_decay` attribute set on Parameters (e.g. A_log, D,
    UV/σ MLPs in SurgicalMamba). The no-decay subgroup gets weight_decay=0.
    """
    if isinstance(group, dict):
        params = list(group["params"])
        base = {k: v for k, v in group.items() if k != "params"}
    else:
        params = list(group)
        base = {}
    decay    = [p for p in params if not getattr(p, "_no_weight_decay", False)]
    no_decay = [p for p in params if     getattr(p, "_no_weight_decay", False)]
    out = []
    if decay:
        out.append({**base, "params": decay,    "weight_decay": wd})
    if no_decay:
        out.append({**base, "params": no_decay, "weight_decay": 0.0})
    return out


def build_optimizer(model, cfg):
    lr = cfg.train.lr
    wd = cfg.train.weight_decay
    backbone_factor = cfg.train.get("backbone_lr_factor", 1.0)
    llrd_decay      = cfg.train.get("llrd_decay", None)  # e.g. 0.75; None disables

    is_convnext = any(s.startswith("extractor.backbone.stages.")
                      for s, _ in model.named_parameters())

    if llrd_decay is not None and is_convnext:
        # Layer-wise LR decay across ConvNeXt depths.
        n_stages = len(model.extractor.backbone.stages)        # 4 for tiny
        bb_top_lr = lr * backbone_factor                       # top-of-backbone lr

        depth_groups: dict[int, list] = {}
        other_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            d = _convnext_depth(n, n_stages)
            if d is None:
                other_params.append(p)
            else:
                depth_groups.setdefault(d, []).append(p)

        param_groups = [{"params": other_params, "lr": lr}]
        print(f"[optimizer] base lr={lr:.2e}, backbone_top lr={bb_top_lr:.2e}, "
              f"llrd_decay={llrd_decay}")
        print(f"  non-backbone: lr={lr:.2e}, params={sum(p.numel() for p in other_params):,}")
        for d in sorted(depth_groups):
            depth_lr = bb_top_lr * (llrd_decay ** (n_stages - d))
            param_groups.append({"params": depth_groups[d], "lr": depth_lr})
            print(f"  depth {d:>1}: lr={depth_lr:.2e}, "
                  f"params={sum(p.numel() for p in depth_groups[d]):,}")
    elif backbone_factor != 1.0:
        backbone_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and n.startswith("extractor.")]
        other_params    = [p for n, p in model.named_parameters()
                           if p.requires_grad and not n.startswith("extractor.")]
        param_groups = [
            {"params": other_params,    "lr": lr},
            {"params": backbone_params, "lr": lr * backbone_factor},
        ]
        print(f"[optimizer] backbone lr={lr * backbone_factor:.2e} ({backbone_factor}×), "
              f"other lr={lr:.2e}  "
              f"| backbone params={sum(p.numel() for p in backbone_params):,}, "
              f"other params={sum(p.numel() for p in other_params):,}")
    else:
        param_groups = [[p for p in model.parameters() if p.requires_grad]]

    # Split each group into wd / no-wd subgroups (honors `_no_weight_decay`).
    split_groups = []
    n_no_decay = 0
    for g in param_groups:
        sub = _split_wd(g, wd)
        for s in sub:
            if s["weight_decay"] == 0.0:
                n_no_decay += sum(p.numel() for p in s["params"])
        split_groups.extend(sub)
    print(f"[optimizer] no-weight-decay params: {n_no_decay:,}")
    return torch.optim.AdamW(split_groups, lr=lr, weight_decay=wd)


def build_scheduler(optimizer, cfg, total_steps):
    warmup_steps = cfg.train.warmup_epochs * len(_splits(cfg)["train_videos"])
    if cfg.train.scheduler == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress     = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine       = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr_min_ratio = cfg.train.lr_min / cfg.train.lr
            return lr_min_ratio + (1.0 - lr_min_ratio) * cosine
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return None


# ── Per-video training pass ───────────────────────────────────────────────────

def run_video(model, video_id, cfg, device,
              optimizer=None, scheduler=None, scaler=None,
              is_train=True, use_amp=False, amp_dtype=torch.bfloat16):
    dataset = VideoClipDataset(video_id=video_id, **_dataset_kwargs(cfg, is_train=is_train))
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=cfg.train.num_workers, pin_memory=True)

    num_phases    = cfg.data.num_phases
    total_loss    = 0.0
    total_loss_ce = 0.0
    total_loss_sm = 0.0
    total_loss_tr = 0.0
    total_correct = 0
    total_frames  = 0
    phase_correct = defaultdict(int)
    phase_total   = defaultdict(int)

    tbptt_k      = cfg.train.get("tbptt_k", 1) if is_train else 1
    slow_states      = None
    window_loss  = None
    window_clips = 0
    n_clips      = len(loader)

    if is_train:
        optimizer.zero_grad()

    for clip_idx, (frames, _, labels, valid_mask, _) in enumerate(loader):
        frames     = frames.to(device)
        labels     = labels.to(device)
        valid_mask = valid_mask.to(device)

        with torch.set_grad_enabled(is_train):
            with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits, slow_states, _, lambdas = model.forward_clip(
                    frames, slow_states=slow_states,
                )

                loss_ce = masked_ce_loss(
                    logits, labels, valid_mask,
                    label_smoothing=cfg.train.label_smoothing,
                )
                loss_sm = temporal_smoothness_loss(logits, valid_mask)

                if lambdas:
                    trans_target = gaussian_transition_map(labels)  # (B, T) ∈ [0,1]
                    mask_flat = valid_mask.bool()
                    int_losses = []
                    for lam in lambdas:  # lam: (B, T)
                        int_losses.append(F.binary_cross_entropy_with_logits(
                            lam[mask_flat], trans_target[mask_flat],
                        ))
                    loss_int = sum(int_losses) / len(int_losses)
                else:
                    loss_int = logits.new_tensor(0.0)

                w_intensity = cfg.train.get("w_intens", 0.1)
                loss = loss_ce + cfg.train.w_smooth * loss_sm + w_intensity * loss_int

        if is_train and not torch.isfinite(loss):
            print(f"\n[NaN-GUARD] video={video_id} clip={clip_idx} loss={loss.item()}")
            print(f"  loss_ce={loss_ce.item()}  loss_sm={loss_sm.item()}  "
                  f"loss_int={loss_int.item()}")
            with torch.no_grad():
                lf = torch.isfinite(logits).all().item()
                print(f"  logits finite={lf}  min={logits.min().item():.3e}  "
                      f"max={logits.max().item():.3e}  "
                      f"abs.mean={logits.abs().mean().item():.3e}")
                if not lf:
                    nan_rows = (~torch.isfinite(logits)).any(dim=-1)
                    first_bad_t = nan_rows.nonzero()[0].tolist() if nan_rows.any() else None
                    print(f"  first non-finite (b,t)={first_bad_t}")
                if lambdas:
                    for li, lam in enumerate(lambdas):
                        lamf = torch.isfinite(lam).all().item()
                        print(f"  lambdas[{li}] finite={lamf}  "
                              f"min={lam.min().item():.3e}  max={lam.max().item():.3e}")
                if slow_states is not None:
                    for bi, s in enumerate(slow_states):
                        if s is None:
                            continue
                        names = ("ssm", "conv")
                        for j, t in enumerate(s):
                            if not torch.is_tensor(t):
                                continue
                            fin = torch.isfinite(t.float()).all().item()
                            flag = "" if fin else "  <<NaN>>"
                            name = names[j] if j < len(names) else f"slot{j}"
                            print(f"  slow_states[{bi}].{name} "
                                  f"shape={tuple(t.shape)} finite={fin}"
                                  f"  abs.max={t.float().abs().max().item():.3e}{flag}")
                for n, p in model.named_parameters():
                    if p.requires_grad and not torch.isfinite(p).all():
                        print(f"  param NaN: {n}")
                for n, b in model.named_buffers():
                    if not torch.isfinite(b.float()).all():
                        print(f"  buffer NaN: {n}")
            raise RuntimeError(f"Loss NaN at video={video_id} clip={clip_idx}")

        if is_train:
            window_loss   = loss if window_loss is None else window_loss + loss
            window_clips += 1
            is_last_clip  = (clip_idx == n_clips - 1)
            is_window_end = (window_clips >= tbptt_k) or is_last_clip

            if is_window_end:
                avg_loss = window_loss / window_clips
                if scaler is not None:
                    scaler.scale(avg_loss).backward()
                else:
                    avg_loss.backward()

                with torch.no_grad():
                    bad = [(n, p.grad) for n, p in model.named_parameters()
                           if p.grad is not None and not torch.isfinite(p.grad).all()]
                    if bad:
                        print(f"\n[NaN-GUARD] video={video_id} clip={clip_idx} "
                              f"after backward: {len(bad)} params have non-finite grad")
                        for n, g in bad[:20]:
                            print(f"  grad NaN/Inf: {n}  "
                                  f"abs.max={g.float().abs().max().item():.3e}")
                        raise RuntimeError(f"Grad NaN at video={video_id} clip={clip_idx}")

                if slow_states is not None:
                    slow_states = [
                        tuple(s_i.detach() for s_i in s) if s is not None else None
                        for s in slow_states
                    ]
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

                window_loss  = None
                window_clips = 0

        total_loss    += loss.item()
        total_loss_ce += loss_ce.item()
        total_loss_sm += loss_sm.item()
        total_loss_tr += loss_int.item()
        correct, total = compute_accuracy(logits, labels, valid_mask)
        total_correct += correct
        total_frames  += total
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            for ph in range(num_phases):
                ph_mask = valid_mask & (labels == ph)
                phase_total[ph]   += ph_mask.sum().item()
                phase_correct[ph] += (preds[ph_mask] == ph).sum().item()

    if is_train and scheduler is not None:
        scheduler.step()

    nc = max(1, len(loader))
    return {
        "loss":       total_loss    / nc,
        "loss_ce":    total_loss_ce / nc,
        "loss_smooth": total_loss_sm / nc,
        "loss_trans":  total_loss_tr / nc,
        "correct":    total_correct,
        "total":      total_frames,
        "phase_correct": phase_correct,
        "phase_total":   phase_total,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # ── Data ──
    p.add_argument("--dataset", required=True, choices=list(DATASET_SPLITS))
    p.add_argument("--data_root", required=True)
    p.add_argument("--phase_annotation_dir", required=True)
    p.add_argument("--tool_annotation_dir", default="_no_tools")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--seq_len", type=int, default=128)

    # ── Model ──
    p.add_argument("--backbone", default="convnext_tiny")
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--d_state", type=int, default=64)
    p.add_argument("--d_state_slow", type=int, default=64)
    p.add_argument("--d_conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--t_rank_block", type=int, default=16)
    p.add_argument("--chunk_size_block", type=int, default=64)
    p.add_argument("--chunk_size_fast_block", type=int, default=64)
    p.add_argument("--chunk_size_slow_block", type=int, default=64)
    p.add_argument("--head_layers", type=int, default=1)
    p.add_argument("--head_chunk_size", type=int, default=64)
    p.add_argument("--freeze_backbone", action="store_true", default=True)
    p.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    p.add_argument("--backbone_trainable_stages", type=int, default=2)
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--output_dropout", type=float, default=0.1)
    p.add_argument("--mamba_dropout", type=float, default=0.1)

    # ── Train ──
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="use_amp", action="store_false")
    p.add_argument("--amp_dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--backbone_lr_factor", type=float, default=0.5)
    p.add_argument("--llrd_decay", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--tbptt_k", type=int, default=6)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--w_smooth", type=float, default=1.0)
    p.add_argument("--w_intens", type=float, default=1.0)
    p.add_argument("--scheduler", default="cosine")
    p.add_argument("--save_dir", default="./checkpoints")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--wandb_project", default="surgical-phase-mamba")
    p.add_argument("--wandb_run_name", default=None)
    return p


def _args_to_cfg(args) -> "OmegaConf":
    return OmegaConf.create({
        "data": {
            "dataset":              args.dataset,
            "data_root":            args.data_root,
            "phase_annotation_dir": args.phase_annotation_dir,
            "tool_annotation_dir":  args.tool_annotation_dir,
            "img_size":             args.img_size,
            "seq_len":              args.seq_len,
            "num_phases":           DATASET_SPLITS[args.dataset]["num_phases"],
        },
        "model": {
            "backbone":                  args.backbone,
            "d_model":                   args.d_model,
            "n_layers":                  args.n_layers,
            "d_state":                   args.d_state,
            "d_state_slow":              args.d_state_slow,
            "d_conv":                    args.d_conv,
            "expand":                    args.expand,
            "t_rank_block":              args.t_rank_block,
            "chunk_size_block":          args.chunk_size_block,
            "chunk_size_fast_block":     args.chunk_size_fast_block,
            "chunk_size_slow_block":     args.chunk_size_slow_block,
            "head_layers":               args.head_layers,
            "head_chunk_size":           args.head_chunk_size,
            "freeze_backbone":           args.freeze_backbone,
            "backbone_trainable_stages": args.backbone_trainable_stages,
            "grad_checkpointing":        args.grad_checkpointing,
            "output_dropout":            args.output_dropout,
            "mamba_dropout":             args.mamba_dropout,
        },
        "train": {
            "seed":               args.seed,
            "device":             args.device,
            "num_workers":        args.num_workers,
            "epochs":             args.epochs,
            "warmup_epochs":      args.warmup_epochs,
            "use_amp":            args.use_amp,
            "amp_dtype":          args.amp_dtype,
            "lr":                 args.lr,
            "lr_min":             args.lr_min,
            "backbone_lr_factor": args.backbone_lr_factor,
            "llrd_decay":         args.llrd_decay,
            "weight_decay":       args.weight_decay,
            "grad_clip":          args.grad_clip,
            "tbptt_k":            args.tbptt_k,
            "label_smoothing":    args.label_smoothing,
            "w_smooth":           args.w_smooth,
            "w_intens":           args.w_intens,
            "scheduler":          args.scheduler,
            "save_dir":           args.save_dir,
            "log_interval":       args.log_interval,
            "patience":           args.patience,
            "wandb_project":      args.wandb_project,
            "wandb_run_name":     args.wandb_run_name,
        },
    })


def main():
    args = _build_parser().parse_args()
    cfg = _args_to_cfg(args)

    torch.manual_seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    model = CausalSurgicalMamba.from_config(cfg).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")
    for name, module in model.named_children():
        t = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if t > 0:
            print(f"  [trainable] {name}: {t:,}")

    train_videos_full = _splits(cfg)["train_videos"]
    optimizer   = build_optimizer(model, cfg)
    total_steps = cfg.train.epochs * len(train_videos_full)
    scheduler   = build_scheduler(optimizer, cfg, total_steps)

    use_amp   = device.type == "cuda" and cfg.train.get("use_amp", True)
    amp_dtype = torch.bfloat16 if cfg.train.get("amp_dtype", "bfloat16") == "bfloat16" \
                else torch.float16
    scaler    = GradScaler("cuda") if (use_amp and amp_dtype == torch.float16) else None
    print(f"Mixed precision: {'enabled' if use_amp else 'disabled'} "
          f"[dtype={amp_dtype}, scaler={'on' if scaler else 'off'}]")

    import time
    save_tag = os.path.basename(os.path.normpath(cfg.train.save_dir))
    default_run = f"{save_tag}_d{cfg.model.get('d_model', 768)}_{time.strftime('%m%d-%H%M')}"
    run_name = cfg.train.get("wandb_run_name", default_run)
    project  = cfg.train.get("wandb_project", "surgical-phase-mamba")
    wandb.init(
        project=project,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name,
    )

    os.makedirs(cfg.train.save_dir, exist_ok=True)
    best_val_acc = -float("inf")
    patience = cfg.train.get("patience", 0)
    epochs_no_improve = 0

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        epoch_loss = epoch_loss_ce = epoch_loss_sm = epoch_loss_tr = 0.0
        epoch_correct = epoch_total = 0
        epoch_phase_correct = defaultdict(int)
        epoch_phase_total   = defaultdict(int)

        train_videos = list(_splits(cfg)["train_videos"])
        random.shuffle(train_videos)

        for step, video_id in enumerate(train_videos, 1):
            result = run_video(
                model, video_id, cfg, device,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                is_train=True, use_amp=use_amp, amp_dtype=amp_dtype,
            )
            epoch_loss    += result["loss"]
            epoch_loss_ce += result["loss_ce"]
            epoch_loss_sm += result["loss_smooth"]
            epoch_loss_tr += result["loss_trans"]
            epoch_correct += result["correct"]
            epoch_total   += result["total"]
            for ph in range(cfg.data.num_phases):
                epoch_phase_correct[ph] += result["phase_correct"][ph]
                epoch_phase_total[ph]   += result["phase_total"][ph]

            if step % cfg.train.log_interval == 0 or step == len(train_videos):
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch} | video {video_id:02d} "
                      f"| loss {result['loss']:.4f} | lr {lr_now:.2e}")
                wandb.log({
                    "train/video_loss": result["loss"],
                    "train/lr": lr_now,
                    "epoch": epoch,
                    "step": (epoch - 1) * len(train_videos) + step,
                })

        n_vid         = len(train_videos)
        train_acc     = epoch_correct / max(1, epoch_total)
        train_loss    = epoch_loss    / n_vid
        train_loss_ce = epoch_loss_ce / n_vid
        train_loss_sm = epoch_loss_sm / n_vid
        train_loss_tr = epoch_loss_tr / n_vid

        log_dict = {
            "train/loss":            train_loss,
            "train/loss_ce":         train_loss_ce,
            "train/loss_smooth":     train_loss_sm,
            "train/loss_intensity":  train_loss_tr,
            "train/accuracy":        train_acc,
            "epoch":                 epoch,
        }
        print(f"\n=== Epoch {epoch} Train | loss {train_loss:.4f} "
              f"(ce={train_loss_ce:.3f} sm={train_loss_sm:.3f} int={train_loss_tr:.3f})"
              f" | acc {train_acc:.4f} ===")
        wandb.log(log_dict)

        mo = evaluate_val(model, cfg, device, epoch)
        model.train()

        score = mo["acc"]
        if score > best_val_acc:
            best_val_acc = score
            epochs_no_improve = 0
            ckpt_path = os.path.join(cfg.train.save_dir, "best_causal.pt")
            torch.save({"model": model.state_dict()}, ckpt_path)
            print(f"  Saved best → {ckpt_path} (val acc={mo['acc']:.4f})")
        else:
            epochs_no_improve += 1

        torch.save({"model": model.state_dict()},
                   os.path.join(cfg.train.save_dir, "latest_causal.pt"))

        if patience > 0 and epochs_no_improve >= patience:
            print(f"\n[EARLY STOP] no val improvement for {patience} epochs "
                  f"(best score={best_val_acc:.4f}). Halting at epoch {epoch}.")
            break

    # ── Final test evaluation on best (val-selected) checkpoint ───────────────
    best_path = os.path.join(cfg.train.save_dir, "best_causal.pt")
    if os.path.exists(best_path):
        print(f"\nLoading best checkpoint for final test evaluation: {best_path}")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        test_mc = evaluate_test(model, cfg, device, epoch=0)
        print(f"[FINAL TEST] test_acc={test_mc['acc']:.4f}")

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()

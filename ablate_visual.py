"""
Visual Token Ablation Study
============================
Compares model accuracy with/without visual tokens to verify they are
actually used by the LLM.

Usage:
    conda run -n surgery_mamba python ablate_visual.py --ckpt checkpoints/best.pt
    conda run -n surgery_mamba python ablate_visual.py --ckpt checkpoints/best.pt --videos 41 42 43

Options:
    --ckpt     Path to checkpoint (.pt file with 'model_state_dict' key, or raw state_dict)
    --videos   Space-separated video IDs to evaluate (default: val_videos from configs.yaml)
    --config   Path to configs.yaml (default: ./configs.yaml next to this script)
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from models import SurgicalPhaseLLM
from data.dataset import VideoClipDataset


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_cls_metrics(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    acc = (preds == labels).float().mean().item()
    per_cls_prec, per_cls_rec, per_cls_jac = [], [], []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().float()
        fp = ((preds == c) & (labels != c)).sum().float()
        fn = ((preds != c) & (labels == c)).sum().float()
        prec = (tp / (tp + fp + 1e-8)).item()
        rec  = (tp / (tp + fn + 1e-8)).item()
        jac  = (tp / (tp + fp + fn + 1e-8)).item()
        per_cls_prec.append(prec)
        per_cls_rec.append(rec)
        per_cls_jac.append(jac)
    return {
        "acc":       acc,
        "precision": sum(per_cls_prec) / num_classes,
        "recall":    sum(per_cls_rec)  / num_classes,
        "jaccard":   sum(per_cls_jac)  / num_classes,
    }


def temporal_smooth_logits(logits: torch.Tensor, window: int = 15) -> torch.Tensor:
    probs   = torch.softmax(logits, dim=-1)
    probs_t = probs.T.unsqueeze(0)
    pad     = window // 2
    smoothed = nn.functional.avg_pool1d(
        probs_t, kernel_size=window, stride=1, padding=pad
    )
    return smoothed.squeeze(0).T


# ── Per-video inference ───────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, video_id, cfg, device, ablate_visual=False):
    dataset = VideoClipDataset(
        video_id  = video_id,
        data_root = cfg.data.data_root,
        phase_dir = cfg.data.phase_annotation_dir,
        tool_dir  = cfg.data.tool_annotation_dir,
        seq_len   = cfg.data.seq_len,
        img_size  = cfg.data.img_size,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=cfg.train.num_workers, pin_memory=True)

    prompt_kv   = model.build_prompt_kv()
    memory      = None
    prev_visual = None
    all_logits  = []
    all_labels  = []

    for frames, tools, labels, valid_mask in loader:
        frames     = frames.to(device)
        tools      = tools.to(device)
        labels     = labels.to(device)
        valid_mask = valid_mask.to(device)

        logits, memory, prev_visual, _, _ = model.forward_clip(
            frames        = frames,
            tool_annots   = tools,
            memory        = memory,
            prev_visual   = prev_visual,
            prompt_kv     = prompt_kv,
            ablate_visual = ablate_visual,
        )

        mask_sq  = valid_mask[0]
        all_logits.append(logits[0][mask_sq].cpu())
        all_labels.append(labels[0][mask_sq].cpu())

    return torch.cat(all_logits), torch.cat(all_labels)


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate_videos(model, video_ids, cfg, device, ablate_visual, label):
    smooth_window = cfg.train.get("test_smooth_window", 15)
    min_seg_len   = cfg.train.get("test_min_segment",   10)
    num_phases    = cfg.data.num_phases

    from train import remove_short_segments

    video_metrics = []
    for vid in video_ids:
        print(f"  [{label}] video {vid:02d} ...", end=" ", flush=True)
        logits, labels = run_inference(model, vid, cfg, device, ablate_visual)
        smoothed = temporal_smooth_logits(logits, window=smooth_window)
        preds    = smoothed.argmax(dim=-1)
        preds    = remove_short_segments(preds, min_len=min_seg_len)
        m = compute_cls_metrics(preds, labels, num_phases)
        video_metrics.append(m)
        print(f"acc={m['acc']:.4f}")

    avg = {k: sum(v[k] for v in video_metrics) / len(video_metrics)
           for k in ["acc", "precision", "recall", "jaccard"]}
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   required=True, help="Checkpoint path")
    parser.add_argument("--videos", nargs="+", type=int, default=None,
                        help="Video IDs to evaluate (default: val_videos from config)")
    parser.add_argument("--config", default=None, help="Path to configs.yaml")
    args = parser.parse_args()

    config_path = args.config or os.path.join(os.path.dirname(__file__), "configs.yaml")
    cfg    = OmegaConf.load(config_path)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading checkpoint from {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # Use config embedded in checkpoint if present (ensures backbone match)
    if "cfg" in ckpt:
        cfg = OmegaConf.merge(cfg, ckpt["cfg"])
        print(f"  Using checkpoint cfg: visual_backbone={cfg.model.visual_backbone}")

    model = SurgicalPhaseLLM.from_config(cfg).to(device)

    # Support both 'model' and 'model_state_dict' keys
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys  : {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    model.eval()

    video_ids = args.videos or list(cfg.data.val_videos)
    print(f"\nEvaluating on videos: {video_ids}\n")

    # ── Normal inference ──────────────────────────────────────────────────────
    normal = evaluate_videos(model, video_ids, cfg, device,
                             ablate_visual=False, label="NORMAL ")

    # ── Ablated inference (visual tokens → zeros) ─────────────────────────────
    ablated = evaluate_videos(model, video_ids, cfg, device,
                              ablate_visual=True,  label="ABLATED")

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Metric':<14} {'Normal':>10} {'Ablated (vis=0)':>16} {'Δ (N-A)':>10}")
    print("-" * 60)
    for k in ["acc", "precision", "recall", "jaccard"]:
        delta = normal[k] - ablated[k]
        print(f"{k:<14} {normal[k]:>10.4f} {ablated[k]:>16.4f} {delta:>+10.4f}")
    print("=" * 60)

    if normal["acc"] - ablated["acc"] > 0.02:
        print("\n✓ Visual tokens ARE being used by the LLM "
              f"(+{(normal['acc'] - ablated['acc'])*100:.1f}% acc drop when zeroed).")
    elif normal["acc"] - ablated["acc"] > 0:
        print("\n~ Marginal visual token contribution "
              f"({(normal['acc'] - ablated['acc'])*100:.1f}% acc drop).")
    else:
        print("\n✗ Visual tokens do NOT seem to contribute "
              "(no accuracy drop when zeroed — potential alignment issue).")


if __name__ == "__main__":
    main()

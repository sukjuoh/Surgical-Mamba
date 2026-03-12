"""
Component Ablation Study
=========================
Measures the contribution of each LLM input component by zeroing it out
and comparing accuracy against the full model.

Components tested:
  visual   — per-frame visual feature tokens (VMamba → reprogramming → projector)
  hints    — clip-level summary tokens (ClipHintEncoder)
  tool     — per-segment tool text embeddings
  memory   — cross-clip global memory tokens
  prev     — local context (previous clip visual tokens, compressed)
  prompt   — static task description prefix

Usage:
    conda run -n surgery_mamba python ablate_visual.py --ckpt checkpoints/best.pt
    conda run -n surgery_mamba python ablate_visual.py --ckpt checkpoints/best.pt --videos 41 42 43
"""

import argparse
import os
import sys

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from models import SurgicalPhaseLLM
from data.dataset import VideoClipDataset


# ── Metrics ───────────────────────────────────────────────────────────────────

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


# ── Per-video inference ────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, video_id, cfg, device, ablate_flags: dict):
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

    prompt_kv   = None if ablate_flags.get("ablate_prompt") else model.build_prompt_kv()
    memory      = None
    prev_visual = None
    all_logits  = []
    all_labels  = []

    for frames, tools, labels, valid_mask, frame_start in loader:
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
            **ablate_flags,
        )
        memory = memory.detach() if memory is not None else None

        mask_sq = valid_mask[0]
        all_logits.append(logits[0][mask_sq].cpu())
        all_labels.append(labels[0][mask_sq].cpu())

    return torch.cat(all_logits), torch.cat(all_labels)


def evaluate(model, video_ids, cfg, device, ablate_flags: dict, label: str) -> dict:
    num_phases    = cfg.data.num_phases
    video_metrics = []
    for vid in video_ids:
        print(f"  [{label:<14}] video {vid:02d} ...", end=" ", flush=True)
        logits, labels = run_inference(model, vid, cfg, device, ablate_flags)
        preds = logits.argmax(dim=-1)
        m = compute_cls_metrics(preds, labels, num_phases)
        video_metrics.append(m)
        print(f"acc={m['acc']:.4f}")
    return {k: sum(v[k] for v in video_metrics) / len(video_metrics)
            for k in ["acc", "precision", "recall", "jaccard"]}


# ── Ablation conditions ────────────────────────────────────────────────────────

# Each entry: (display label, ablate_flags dict)
# ablate_flags are passed directly as kwargs to forward_clip.
# "no memory" and "no prev" are achieved by never passing those tensors;
# the corresponding flags tell forward_clip to ignore them even if present.
ABLATION_CONDITIONS = [
    ("FULL (baseline)",  {}),
    ("no visual",        {"ablate_visual": True}),
    ("no hints",         {"ablate_hints":  True}),
    ("no tool text",     {"ablate_tool":   True}),
    ("no memory",        {"ablate_memory": True}),
    ("no prev clip",     {"ablate_prev":   True}),
    ("no prompt",        {"ablate_prompt": True}),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   required=True, help="Checkpoint path")
    parser.add_argument("--videos", nargs="+", type=int, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config_path = args.config or os.path.join(os.path.dirname(__file__), "configs.yaml")
    cfg    = OmegaConf.load(config_path)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint from {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "cfg" in ckpt:
        cfg = OmegaConf.merge(cfg, ckpt["cfg"])
        print(f"  checkpoint cfg: visual_backbone={cfg.model.visual_backbone}")

    model = SurgicalPhaseLLM.from_config(cfg).to(device)
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing    : {missing}")
    if unexpected:
        print(f"  Unexpected : {unexpected}")
    model.eval()

    video_ids = args.videos or list(cfg.data.val_videos)
    print(f"\nEvaluating on videos: {video_ids}\n")

    results = {}
    for label, flags in ABLATION_CONDITIONS:
        print(f"\n── {label} ──")
        results[label] = evaluate(model, video_ids, cfg, device, flags, label)

    # ── Report ────────────────────────────────────────────────────────────────
    baseline_acc = results["FULL (baseline)"]["acc"]

    print("\n" + "=" * 72)
    print(f"{'Condition':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'Jac':>8} {'ΔAcc':>8}")
    print("-" * 72)
    for label, _ in ABLATION_CONDITIONS:
        m = results[label]
        delta = m["acc"] - baseline_acc
        flag  = "  (baseline)" if label == "FULL (baseline)" else f"  {delta:+.4f}"
        print(f"{label:<20} {m['acc']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['jaccard']:>8.4f}{flag}")
    print("=" * 72)

    print("\nInterpretation (ΔAcc = ablated − baseline, negative = component helps):")
    for label, _ in ABLATION_CONDITIONS[1:]:
        delta = results[label]["acc"] - baseline_acc
        pct   = abs(delta) * 100
        if delta < -0.02:
            verdict = f"✓  contributes  ({pct:.1f}% drop)"
        elif delta < 0:
            verdict = f"~  marginal     ({pct:.1f}% drop)"
        else:
            verdict = f"✗  not used     ({pct:.1f}% change)"
        print(f"  {label:<18}: {verdict}")


if __name__ == "__main__":
    main()

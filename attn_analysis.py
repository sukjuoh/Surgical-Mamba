"""
LLM Attention Analysis
=======================
Visualizes how much attention visual tokens give to each input region
(prompt, memory, hints, tool text, prev clip, visual tokens themselves).

For each LLM layer, plots a heatmap:
  rows    = visual token index (frame 0 … T-1)
  columns = input regions (prompt / memory / hints / tool / prev / visual_hdr / visual_self)

Also plots a summary bar chart: average attention from all visual tokens to each region,
averaged over all layers and all heads.

Usage:
    conda run -n surgery_mamba python attn_analysis.py \\
        --ckpt checkpoints/best.pt --video 41 --out attn_figs/

Options:
    --ckpt    checkpoint path
    --video   video ID to evaluate (all clips)
    --layers  LLM layer indices for per-layer bar charts (default: first, middle, last)
    --out     output directory for figures (default: ./attn_figs)
    --config  path to configs.yaml
"""

import argparse
import os
import sys

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from models import SurgicalPhaseLLM
from data.dataset import VideoClipDataset


# ── Region grouping ───────────────────────────────────────────────────────────
# Headers are short; merge them with their payload for readability.
_REGION_GROUPS_BASE = [
    ("Prompt",      ["prompt"]),
    ("Memory",      ["memory_hdr", "memory"]),
    ("Hints",       ["hints_hdr", "hints"]),
    ("Prev clip",   ["prev_hdr", "prev"]),
    ("Tool tokens", ["tool_hdr", "tool"]),
    ("Visual hdr",  ["visual_hdr"]),
    ("Visual self", ["visual"]),
]


def get_region_groups(use_prev_clip: bool = True) -> list:
    if use_prev_clip:
        return _REGION_GROUPS_BASE
    return [g for g in _REGION_GROUPS_BASE if g[0] != "Prev clip"]

# Colours for bar chart
_COLOURS = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#64b5cd", "#e07b39"]


def build_region_labels(regions: dict, total_len: int, region_groups: list) -> list:
    """Return a per-token list of region group names for axis tick labelling."""
    labels = ["?"] * total_len
    for group_name, keys in region_groups:
        for k in keys:
            if k not in regions:
                continue
            s, e = regions[k]
            for i in range(s, min(e, total_len)):
                labels[i] = group_name
    return labels


def aggregate_by_region(attn_row: np.ndarray, regions: dict, total_len: int, region_groups: list) -> dict:
    """
    Sum attention weights in attn_row (shape: total_len) into region groups.
    Returns dict group_name → summed attention.
    """
    out = {}
    for group_name, keys in region_groups:
        total = 0.0
        for k in keys:
            if k not in regions:
                continue
            s, e = regions[k]
            e = min(e, total_len)
            if s < total_len:
                total += float(attn_row[s:e].sum())
        out[group_name] = total
    return out


@torch.no_grad()
def extract_attentions_video(model, video_id, cfg, device):
    """
    Run all clips of a video and accumulate per-layer, per-region attention sums.

    Returns:
        region_sums:  dict  layer_idx → dict  group_name → float  (sum over all clips/frames/heads)
        region_counts: int  total (clips × T × heads) used for averaging
        regions:      dict  region name → (start, end) — from the last clip (representative)
        T:            int   frames per clip
        num_layers:   int
        region_groups: list  active region groups (respects model.use_prev_clip)
    """
    dataset = VideoClipDataset(
        video_id  = video_id,
        data_root = cfg.data.data_root,
        phase_dir = cfg.data.phase_annotation_dir,
        tool_dir  = cfg.data.tool_annotation_dir,
        seq_len   = cfg.data.seq_len,
        img_size  = cfg.data.img_size,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=0, pin_memory=False)

    region_groups = get_region_groups(model.use_prev_clip)

    prompt_kv   = model.build_prompt_kv()
    memory      = None
    prev_visual = None
    num_layers  = model.llm.config.num_hidden_layers

    # layer → group → accumulated sum
    region_sums = {li: {g: 0.0 for g, _ in region_groups} for li in range(num_layers)}
    total_counts = 0   # number of (frame × head) samples accumulated
    last_regions = None
    T = cfg.data.seq_len
    n_clips = len(loader)

    for i, (frames, tools, labels, valid_mask, frame_start) in enumerate(loader):
        frames = frames.to(device)
        tools  = tools.to(device)
        print(f"  clip {i+1:3d}/{n_clips}", end="\r", flush=True)

        out = model.forward_clip(
            frames=frames, tool_annots=tools,
            memory=memory, prev_visual=prev_visual, prompt_kv=prompt_kv,
            output_attentions=True,
        )
        _, _, memory, prev_visual, _, _, attentions, regions = out
        memory = memory.detach() if memory is not None else None
        last_regions = regions
        T_actual = frames.shape[1]
        total_len = attentions[0].shape[-1]

        # Accumulate: for each layer, sum attention from visual rows to each region
        prompt_len_offset = regions.get("prompt", (0, 0))[1]
        for li in range(num_layers):
            a = attentions[li][0].float().cpu().numpy()   # (H, seq, seq)
            H = a.shape[0]
            # Determine row offset:
            #   KV-cache mode:      a.shape = (H, llm_input_len, prompt+llm_input_len)
            #                       → rows are 0-indexed over llm_input only
            #                       → subtract prompt_len to convert shifted positions to row indices
            #   Full-sequence mode (LoRA / no KV-cache):
            #                       a.shape = (H, prompt+llm_input_len, prompt+llm_input_len)
            #                       → rows include prompt prefix
            #                       → shifted positions are already correct row indices (row_offset=0)
            row_offset = 0 if a.shape[1] == total_len else prompt_len_offset
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
                # Normalize: divide by H*T so each clip contributes equally
                region_sums[li][group_name] += col_sum / (H * T_actual)

        total_counts += 1   # one clip

    print()  # newline after \r progress
    return region_sums, total_counts, last_regions, T, num_layers, region_groups


def plot_summary_bar(avg_by_region: dict, video_id: int, out_dir: str, suffix: str = "",
                     region_groups: list = None):
    """Bar chart: average attention from visual tokens to each region."""
    if region_groups is None:
        region_groups = _REGION_GROUPS_BASE
    group_names = [g for g, _ in region_groups if avg_by_region.get(g, 0) > 0]
    values      = [avg_by_region[g] for g in group_names]
    label       = "latter-half layers" if suffix == "_latter" else "all layers"

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(group_names, values, color=_COLOURS[:len(group_names)], edgecolor="k", linewidth=0.5)
    ax.set_ylabel("Mean attention weight (avg over clips, heads, frames)")
    ax.set_title(f"Visual token → region attention  |  video {video_id:02d}  ({label})")
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f"attn_bar_v{video_id:02d}{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_layer_bar(region_sums_layer: dict, n_clips: int, layer_idx: int,
                   video_id: int, out_dir: str, region_groups: list = None):
    """Bar chart per layer: avg attention from visual tokens to each region."""
    if region_groups is None:
        region_groups = _REGION_GROUPS_BASE
    group_names = [g for g, _ in region_groups if region_sums_layer.get(g, 0) > 0]
    values      = [region_sums_layer[g] / n_clips for g in group_names]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(group_names, values, color=_COLOURS[:len(group_names)], edgecolor="k", linewidth=0.5)
    ax.set_ylabel("Mean attention weight (avg over clips, heads, frames)")
    ax.set_title(f"Visual token → region attention  |  Layer {layer_idx}  |  video {video_id:02d}")
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f"attn_layer_v{video_id:02d}_L{layer_idx:02d}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   required=True)
    parser.add_argument("--video",  type=int, default=41)
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="LLM layer indices for per-layer bar charts (default: first, middle, last)")
    parser.add_argument("--out",    default="./attn_figs")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    config_path = args.config or os.path.join(os.path.dirname(__file__), "configs.yaml")
    cfg    = OmegaConf.load(config_path)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "cfg" in ckpt:
        cfg = OmegaConf.merge(cfg, ckpt["cfg"])

    model = SurgicalPhaseLLM.from_config(cfg).to(device)
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # sdpa attention does not support output_attentions=True; switch to eager.
    model.llm.set_attn_implementation("eager")

    num_layers = model.llm.config.num_hidden_layers
    if args.layers is None:
        args.layers = sorted({0, num_layers // 2, num_layers - 1})
    print(f"LLM has {num_layers} layers. Per-layer plots: {args.layers}")

    print(f"\nProcessing full video {args.video:02d} ...")
    print(f"  use_prev_clip: {model.use_prev_clip}")
    region_sums, n_clips, regions, T, _, region_groups = extract_attentions_video(
        model, args.video, cfg, device
    )
    print(f"  Total clips processed: {n_clips}")
    print(f"  Regions detected: {list(regions.keys())}")

    # ── All-layer average ─────────────────────────────────────────────────────
    avg_by_region = {g: 0.0 for g, _ in region_groups}
    for li in range(num_layers):
        for g in avg_by_region:
            avg_by_region[g] += region_sums[li][g]
    for g in avg_by_region:
        avg_by_region[g] /= (num_layers * n_clips)

    print("\nAvg attention (ALL layers):")
    max_v = max(avg_by_region.values()) or 1.0
    for g, v in avg_by_region.items():
        bar = "█" * int(v / max_v * 40)
        print(f"  {g:<14}: {v:.4f}  {bar}")

    plot_summary_bar(avg_by_region, args.video, args.out, region_groups=region_groups)

    # ── Latter-half layers average (more meaningful — output head uses last layer) ──
    latter_start = num_layers // 2
    avg_latter = {g: 0.0 for g, _ in region_groups}
    latter_count = num_layers - latter_start
    for li in range(latter_start, num_layers):
        for g in avg_latter:
            avg_latter[g] += region_sums[li][g]
    for g in avg_latter:
        avg_latter[g] /= (latter_count * n_clips)

    print(f"\nAvg attention (latter half: layers {latter_start}~{num_layers-1}):")
    max_v = max(avg_latter.values()) or 1.0
    for g, v in avg_latter.items():
        bar = "█" * int(v / max_v * 40)
        print(f"  {g:<14}: {v:.4f}  {bar}")

    plot_summary_bar(avg_latter, args.video, args.out, suffix="_latter", region_groups=region_groups)

    # ── Per-layer bar charts ──────────────────────────────────────────────────
    for li in args.layers:
        if li < 0 or li >= num_layers:
            print(f"  Warning: layer {li} out of range, skipping.")
            continue
        plot_layer_bar(region_sums[li], n_clips, li, args.video, args.out, region_groups=region_groups)

    print(f"\nDone. Figures saved to: {args.out}/")


if __name__ == "__main__":
    main()

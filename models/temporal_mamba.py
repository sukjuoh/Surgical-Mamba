"""
Temporal refinement for 1D visual token sequences.

MultiScaleCausalConvBlock replaces the single-scale CausalConvBlock.

Design — three parallel dilated depthwise convolutions:
  dilation=1, k=7 →  7-frame receptive field  (immediate neighbourhood)
  dilation=3, k=7 → 19-frame receptive field  (short-term dynamics)
  dilation=7, k=7 → 43-frame receptive field  (mid-term context)

  Outputs are summed then mixed via pointwise conv.
  A single kernel=15 block covers only one scale; parallel dilated convs
  cover three scales at essentially the same FLOPs (all depthwise/grouped).

Frame-delta injection (TemporalRefiner):
  Δx[t] = x[t] - x[t-1] projected and added BEFORE the conv blocks.
  Kept purely additive — not used as a multiplicative gate — to avoid
  amplifying clip-boundary delta artifacts during online inference.

Online inference buffers (per scale, per block):
  scale 1: buffer of 6  past frames  (= (7-1)*1)
  scale 2: buffer of 18 past frames  (= (7-1)*3)
  scale 3: buffer of 42 past frames  (= (7-1)*7)
  Prepend buffer → conv → take last position. Zero train/inference gap.

# ── Previously: single-scale CausalConvBlock (kernel_size=15) ────────────────
# Replaced: fixed 15-frame field too coarse; could not simultaneously
# capture immediate motion and multi-second trends.
# ─────────────────────────────────────────────────────────────────────────────

# ── Previously: Bidirectional Mamba ──────────────────────────────────────────
# Replaced: bidirectional SSM requires full sequence at inference
# (train/inference gap for online use) and is slower than depthwise conv.
# ─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba  # still needed for CrossClipMemory


class MultiScaleCausalConvBlock(nn.Module):
    """
    Multi-scale causal depthwise conv block: 3×DWConv(dilated) → sum → Pointwise → FFN.

    Three parallel dilated depthwise convolutions with kernel=7:
      (dilation=1) → 7-frame receptive field
      (dilation=3) → 19-frame receptive field
      (dilation=7) → 43-frame receptive field

    Causal padding per scale: left-pad by (kernel-1)*dilation, trim right to T.
    Outputs are summed (same shape, no extra mixing) then passed through
    a pointwise linear and FFN with residuals.

    All convolutions are depthwise (groups=d_model) — negligible FLOPs vs
    a single kernel=15 block; parameter count is +2 DWConv weight vectors.

    Online buffer per scale: (kernel-1)*dilation = 6 / 18 / 42 past frames.

    Args:
        d_model: feature dimension (e.g. 768)
    """

    # (kernel_size, dilation) — fixed, no config needed
    _SCALES = [(7, 1), (7, 3), (7, 7)]

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        # Parallel dilated DWConvs — each causal via left-padding
        self.dw_convs = nn.ModuleList([
            nn.Conv1d(
                d_model, d_model, kernel_size=k,
                dilation=d, padding=(k - 1) * d, groups=d_model,
            )
            for k, d in self._SCALES
        ])

        # Pointwise: mix channels after multi-scale aggregation
        self.pw_conv = nn.Linear(d_model, d_model)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        B, T, d = x.shape

        normed = self.norm(x)
        h = normed.transpose(1, 2)                        # (B, d, T)

        # Sum multi-scale outputs; each causal: trim to T
        h = sum(conv(h)[:, :, :T] for conv in self.dw_convs)  # (B, d, T)

        h = h.transpose(1, 2)                             # (B, T, d)
        h = self.pw_conv(h)
        x = x + h

        x = x + self.ffn(self.ffn_norm(x))
        return x


class CrossClipMemory(nn.Module):
    """
    SSM-based cross-clip temporal memory with selective retrieval.

    Each clip:
      1. Cross-attention (hints → prev_memory): current hints query into the
         global memory to selectively retrieve only the past context that is
         relevant to the current clip, rather than blindly consuming all history.
      2. Mamba SSM: [prev_memory | enhanced_hints] → updated memory summary.

    Design:
      - hints (Q) × prev_memory (K/V) → retrieved  (selective read)
      - enhanced_hints = hints + retrieved           (residual fusion)
      - [prev_memory | enhanced_hints] → Mamba → last N positions = new_memory

    Args:
        d_model:  feature dimension (= d_llm)
        n_heads:  number of attention heads for selective retrieval
        d_state:  SSM state size
        d_conv:   depthwise conv kernel size
        expand:   inner dimension expansion factor
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        # Cross-attention: current hints (Q) selectively read from prev_memory (K/V)
        self.q_norm    = nn.LayerNorm(d_model)
        self.kv_norm   = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)

        # Mamba SSM: integrates enhanced hints with full memory history
        self.mamba_norm = nn.LayerNorm(d_model)
        self.mamba      = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.out_norm   = nn.LayerNorm(d_model)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        hints: torch.Tensor,
        prev_memory: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        visual_tokens: (B, T,       d_model) — per-frame projected features (d_llm)
        hints:         (B, N_hints, d_model) — current clip's hint tokens
        prev_memory:   (B, N_hints, d_model) or None — previous clip's memory (detached)

        Two separate paths:

        Read path — memory refreshed by current clip (for LLM context):
            prev_memory(Q) × visual_tokens(K/V) → enriched_memory (B, N_hints, d)
            Each memory slot attends to the current clip and updates itself with
            what it finds relevant. enriched_memory replaces raw prev_memory as
            the LLM's global prefix — the LLM sees a memory tuned to the current
            clip, while visual_tokens remain a clean present-only representation.
            Role separation: memory = past context, visual = current clip.

        Write path — clean per-clip summary (for next clip):
            Mamba([prev_memory | raw hints])[-N:] → new_memory.
            Raw hints keep the write path independent of the read path.

        Returns:
            new_memory:      (B, N_hints, d_model) — stored for next clip (write path)
            enriched_memory: (B, N_hints, d_model) or None — LLM global prefix (read path)
        """
        if prev_memory is not None:
            # Read: each memory slot queries current clip → refreshes itself
            enriched_memory, _ = self.cross_attn(
                self.q_norm(prev_memory),
                self.kv_norm(visual_tokens),
                self.kv_norm(visual_tokens),
            )
            enriched_memory = self.attn_norm(prev_memory + enriched_memory)  # (B, N, d)

            # Write: Mamba over [prev_memory | raw hints]
            x = torch.cat([prev_memory, hints], dim=1)                       # (B, 2N, d)
        else:
            x = hints
            enriched_memory = None

        # Residual SSM
        x = x + self.mamba(self.mamba_norm(x))

        # Last N positions = new memory (for next clip)
        N = hints.shape[1]
        new_memory = self.out_norm(x[:, -N:, :])
        return new_memory, enriched_memory



class LocalContextCompressor(nn.Module):
    """
    Compresses the previous clip's visual tokens to 1/ratio of their temporal length
    via average pooling. No learnable parameters — visual_tokens are already
    well-refined (VMamba + TemporalRefiner + Reprogramming), so averaging suffices.

    (B, T, d_model) → (B, T//ratio, d_model)

    Args:
        ratio:  temporal downsampling factor (default 4 → T//4 output tokens)
    """

    def __init__(self, ratio: int = 4):
        super().__init__()
        self.ratio = ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T//ratio, d_model)"""
        B, T, d = x.shape
        pad = (self.ratio - T % self.ratio) % self.ratio
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        return x.reshape(B, x.shape[1] // self.ratio, self.ratio, d).mean(dim=2)


class TemporalRefiner(nn.Module):
    """
    Stack of N MultiScaleCausalConvBlocks for temporal refinement of visual tokens.

    Prepends a frame-delta injection step: the per-frame visual difference
    x[t] - x[t-1] is projected and added to the input (purely additive —
    not used as a gate, to avoid clip-boundary artifacts in online mode).

    Input:  (B, T, d_model)  — raw VMamba per-frame features
    Output: (B, T, d_model)  — temporally-refined features

    Args:
        d_model:    feature dimension
        num_layers: number of stacked MultiScaleCausalConvBlocks
    """

    def __init__(self, d_model: int, num_layers: int = 2):
        super().__init__()
        # Frame-delta: projects Δx[t] = x[t] - x[t-1] into the feature space
        self.delta_proj = nn.Linear(d_model, d_model, bias=False)

        self.layers = nn.ModuleList([
            MultiScaleCausalConvBlock(d_model)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        # Frame-delta injection: Δx[0] = 0, Δx[t] = x[t] - x[t-1]
        delta = torch.diff(x, dim=1, prepend=x[:, :1, :])
        x = x + self.delta_proj(delta)

        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

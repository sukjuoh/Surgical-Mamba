"""
Temporal refinement for 1D visual token sequences.

CausalConvBlock replaces the previous bidirectional Mamba block.

Why Causal Depthwise Conv:
  - Zero train/inference gap: causal conv behaves identically during
    offline (batch) training and online (frame-by-frame) inference.
    Online step only requires maintaining a rolling buffer of
    (kernel_size - 1) past frames — no SSM state or mode switching.
  - Speed: depthwise conv on (B, 768, T) is microseconds vs Mamba's
    custom CUDA kernel overhead. Pointwise mix adds negligible cost.
  - Local context (kernel_size=15 → 15-frame receptive field) is
    sufficient here because the LLM and CrossClipMemory already model
    long-range temporal dependencies across the full clip and video.
  - Frame-delta injection provides explicit motion signal for transitions,
    complementing the conv's local smoothing.

Online inference buffer:
  At each new frame t, maintain feat_buffer (deque, maxlen=kernel_size-1).
  Prepend buffered frames → conv → take the last output position.
  No retraining needed; behavior is mathematically identical to batch mode.

# ── Previously: Bidirectional Mamba (commented out) ──────────────────────────
# from mamba_ssm import Mamba
#
# class TemporalMambaBlock(nn.Module):
#     def __init__(self, d_model, d_state=16, d_conv=4, expand=2, bidirectional=True):
#         super().__init__()
#         self.bidirectional = bidirectional
#         self.norm = nn.LayerNorm(d_model)
#         self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
#         if bidirectional:
#             self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
#             self.merge = nn.Linear(2 * d_model, d_model, bias=False)
#         self.ffn_norm = nn.LayerNorm(d_model)
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model),
#         )
#     def forward(self, x):
#         normed = self.norm(x)
#         fwd = self.mamba_fwd(normed)
#         if self.bidirectional:
#             rev = torch.flip(self.mamba_bwd(torch.flip(normed, dims=[1])), dims=[1])
#             ssm_out = self.merge(torch.cat([fwd, rev], dim=-1))
#         else:
#             ssm_out = fwd
#         x = x + ssm_out
#         x = x + self.ffn(self.ffn_norm(x))
#         return x
# ─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba  # still needed for CrossClipMemory


class CausalConvBlock(nn.Module):
    """
    Causal Depthwise Conv block: DWConv → Pointwise → FFN, with residuals.

    Causal padding: left-pad by (kernel_size - 1), trim right so each
    output position t sees only frames [t-kernel_size+1 … t].
    Identical in batch training and online (buffer-based) inference.

    Args:
        d_model:     feature dimension (e.g. 768)
        kernel_size: temporal receptive field in frames (default 15)
    """

    def __init__(self, d_model: int, kernel_size: int = 15):
        super().__init__()
        self.kernel_size = kernel_size

        self.norm = nn.LayerNorm(d_model)
        # Depthwise conv: each channel filtered independently (cheap)
        # padding=kernel_size-1 pads left; we trim the extra right positions
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size - 1, groups=d_model,
        )
        # Pointwise: mix channels after spatial filtering
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

        # Depthwise causal conv
        normed = self.norm(x)
        h = normed.transpose(1, 2)                   # (B, d, T)
        h = self.dw_conv(h)[:, :, :T]               # causal: drop right-padded positions
        h = h.transpose(1, 2)                        # (B, T, d)
        h = self.pw_conv(h)
        x = x + h

        # FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x


class CrossClipMemory(nn.Module):
    """
    SSM-based cross-clip temporal memory.

    Each clip, hint tokens are fed through a Mamba layer conditioned on the
    previous clip's memory, producing an updated memory that summarizes all
    clips seen so far. This memory is used as context prefix for the LLM in
    the NEXT clip, replacing the raw hidden-state truncation approach.

    Design:
      - Input: [prev_memory | current_hints] concatenated along sequence dim
      - Mamba processes left-to-right (causal SSM), so current positions attend
        to all previous positions including past-clip memory
      - Output: last N_hints positions = updated memory (residual with hints)

    Args:
        d_model:  feature dimension (= d_llm)
        d_state:  SSM state size
        d_conv:   depthwise conv kernel size
        expand:   inner dimension expansion factor
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, hints: torch.Tensor, prev_memory: torch.Tensor = None) -> torch.Tensor:
        """
        hints:       (B, N_hints, d_model) — current clip's hint tokens
        prev_memory: (B, N_hints, d_model) or None — previous clip's memory (detached)

        Returns:
            memory: (B, N_hints, d_model) — updated memory for next clip
        """
        if prev_memory is not None:
            # Prepend prev_memory so Mamba can condition current hints on past clips
            x = torch.cat([prev_memory, hints], dim=1)  # (B, 2*N, d)
        else:
            x = hints  # (B, N, d)

        # Residual SSM: Mamba integrates temporal context left-to-right
        x = x + self.mamba(self.norm(x))

        # Take the last N_hints positions: updated representation of current clip
        N = hints.shape[1]
        memory = self.out_norm(x[:, -N:, :])  # (B, N_hints, d)
        return memory


class MemoryFusion(nn.Module):
    """
    Cross-attention: memory queries into current visual tokens.

    Past-clip memory acts as queries (Q) to extract relevant information
    from the current clip's visual tokens (K/V). The output (N_hints tokens)
    is prepended to the LLM input — giving the LLM a history-grounded view
    of the current clip, distinct from the raw visual tokens.

        Q = memory         (B, N, d_model)   — past clips summary (detached)
        K = V = visual     (B, T, d_model)   — current clip visual tokens
        Output             (B, N, d_model)   — what current clip offers to past memory

    LLM input: [attended_memory(N) | tool_text | hints | visual_tokens]

    Args:
        d_model:  feature dimension (= d_llm)
        n_heads:  number of attention heads
        dropout:  attention dropout
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.q_norm  = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, memory: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        memory:        (B, N, d_model) — previous clip's memory (detached), used as Q
        visual_tokens: (B, T, d_model) — current clip's visual tokens, used as K/V

        Returns: (B, N, d_model) — memory queries answered by current visual content
        """
        q  = self.q_norm(memory)
        kv = self.kv_norm(visual_tokens)
        attn_out, _ = self.cross_attn(q, kv, kv)
        return self.out_norm(memory + attn_out)


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
    Stack of N CausalConvBlocks for temporal refinement of visual tokens.

    Prepends a frame-delta injection step: the per-frame visual difference
    x[t] - x[t-1] is projected and added to the input, giving each conv block
    an explicit motion signal that highlights phase-transition moments.

    Input:  (B, T, d_model)  — raw VMamba per-frame features
    Output: (B, T, d_model)  — temporally-refined features

    Args:
        d_model:       feature dimension
        num_layers:    number of stacked CausalConvBlocks
        conv_kernel_size: temporal receptive field per block (frames)

    # Previously accepted: d_state, d_conv, expand, bidirectional (Mamba params)
    # These are no longer used and kept only as comments for reference.
    # d_state=16, d_conv=4, expand=2, bidirectional=True
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        conv_kernel_size: int = 15,
    ):
        super().__init__()
        # Frame-delta: projects Δx[t] = x[t] - x[t-1] into the feature space
        self.delta_proj = nn.Linear(d_model, d_model, bias=False)

        self.layers = nn.ModuleList([
            CausalConvBlock(d_model, kernel_size=conv_kernel_size)
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

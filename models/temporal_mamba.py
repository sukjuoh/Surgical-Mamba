"""
Temporal Mamba refiner for 1D visual token sequences.

Wraps mamba_ssm.Mamba to process (B, T, d_model) temporal sequences.
Stacked N layers with pre-norm residual connections, applied BEFORE the
LLM reprogramming step to capture temporal dependencies across frames.

Why Mamba here:
  - Visual tokens from VMamba are spatially rich but temporally independent
    (each frame processed independently by the extractor).
  - A causal SSM refiner lets each frame's representation attend to ALL
    previous frames with O(T) complexity — ideal for surgical video where
    phase identity depends on what happened earlier in the clip.
  - Bidirectional mode: forward + backward Mamba so each token sees the
    full clip context in both directions before being sent to the LLM.

Enhancements over a plain Mamba stack:
  1. True bidirectional scan — forward Mamba + backward Mamba (flipped),
     merged by a linear projection. Each frame sees both past and future.
  2. FFN after each SSM block — post-SSM MLP (pre-norm, GELU) improves
     per-token feature expressiveness, following the Mamba-2 / Jamba recipe.
  3. Frame-delta injection — x[t] - x[t-1] encodes local visual motion,
     which is a strong signal for detecting surgical phase transitions.
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class TemporalMambaBlock(nn.Module):
    """
    Bidirectional Mamba block: SSM (fwd+bwd) → merge → FFN, with residuals.

    Args:
        d_model:       feature dimension (e.g. 768 for VMamba-Tiny)
        d_state:       SSM state size
        d_conv:        depthwise conv kernel size
        expand:        expansion factor for inner dimension
        bidirectional: if True, runs a second backward SSM and merges
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        if bidirectional:
            self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.merge = nn.Linear(2 * d_model, d_model, bias=False)

        # FFN: pre-norm MLP with 4× expansion, following Mamba/Jamba convention
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        normed = self.norm(x)
        fwd = self.mamba_fwd(normed)
        if self.bidirectional:
            rev = torch.flip(self.mamba_bwd(torch.flip(normed, dims=[1])), dims=[1])
            ssm_out = self.merge(torch.cat([fwd, rev], dim=-1))
        else:
            ssm_out = fwd
        x = x + ssm_out
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


class TemporalRefiner(nn.Module):
    """
    Stack of N bidirectional Mamba blocks for temporal refinement of visual tokens.

    Prepends a frame-delta injection step: the per-frame visual difference
    x[t] - x[t-1] is projected and added to the input, giving each SSM block
    an explicit motion signal that highlights phase-transition moments.

    Input:  (B, T, d_model)  — raw VMamba per-frame features
    Output: (B, T, d_model)  — temporally-refined features

    Args:
        d_model:       feature dimension
        num_layers:    number of stacked blocks (1 or 2 recommended)
        d_state:       SSM state size
        d_conv:        depthwise conv kernel size
        expand:        inner expansion factor
        bidirectional: enable backward SSM in each block
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = True,
    ):
        super().__init__()
        # Frame-delta: projects Δx[t] = x[t] - x[t-1] into the feature space
        self.delta_proj = nn.Linear(d_model, d_model, bias=False)

        self.layers = nn.ModuleList([
            TemporalMambaBlock(d_model, d_state, d_conv, expand, bidirectional)
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

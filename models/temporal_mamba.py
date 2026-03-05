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
  - Non-causal (bidirectional) mode: we use bidirectional scan so each
    token can see the full clip context before being sent to the LLM.

Note: Uses Mamba (v1) rather than Mamba2 because causal_conv1d (required
by Mamba2's Triton kernel) is not installed in this environment.
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class TemporalMambaBlock(nn.Module):
    """
    Single Mamba block with pre-LayerNorm and residual connection.

    Args:
        d_model:   feature dimension (e.g. 768 for VMamba-Tiny)
        d_state:   SSM state size
        d_conv:    depthwise conv kernel size
        expand:    expansion factor for inner dimension
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        return x + self.mamba(self.norm(x))


class TemporalRefiner(nn.Module):
    """
    Stack of N Mamba blocks for temporal refinement of visual tokens.

    Input:  (B, T, d_model)  — raw VMamba per-frame features
    Output: (B, T, d_model)  — temporally-refined features

    Args:
        d_model:    feature dimension
        num_layers: number of stacked Mamba blocks (1 or 2 recommended)
        d_state:    SSM state size
        d_conv:     depthwise conv kernel size
        expand:     inner expansion factor
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TemporalMambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

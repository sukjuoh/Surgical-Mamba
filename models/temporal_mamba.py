"""
Temporal refinement for 1D visual token sequences.

BidirectionalMambaBlock:
  Forward Mamba + Backward Mamba, outputs concatenated and projected to d_model.
  Sees the full clip context in both directions — unlike causal conv which is
  limited to a 43-frame receptive field.

  forward scan:  left-to-right SSM  (past → present)
  backward scan: x.flip(1) → SSM → flip  (future → present)
  out = Linear(d_model*2 → d_model) + FFN

TemporalRefiner:
  Stack of BidirectionalMambaBlocks. No frame-delta injection — avoids
  augmentation-induced per-frame noise (random crop/flip differ per frame).
  Mamba SSM naturally captures temporal change via selective state updates.

CrossClipMemory write path — Mamba + GRU-style gating:
  Mamba processes [prev_memory | hints] sequentially (long-range context).
  A GRU-style update gate then explicitly controls per-slot retention:
    candidate = Mamba_out[-N:]          (what Mamba proposes)
    gate = sigmoid(W · [prev_memory, candidate])
    new_memory = gate * candidate + (1 - gate) * prev_memory
  This gives Mamba's expressiveness + GRU's explicit "how much to update" control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba  # still needed for CrossClipMemory


class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional Mamba block: forward SSM + backward SSM over the full sequence.

    Both scans share the same input; their outputs are concatenated and projected
    back to d_model, giving each frame access to full past and future context.

    Args:
        d_model:  feature dimension (e.g. 768)
        d_state:  SSM state size
        d_conv:   depthwise conv kernel inside Mamba
        expand:   inner dimension expansion factor
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.out_proj = nn.Linear(d_model * 2, d_model)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        normed = self.norm(x)
        fwd = self.mamba_fwd(normed)                      # (B, T, d_model)
        bwd = self.mamba_bwd(normed.flip(1)).flip(1)      # (B, T, d_model)
        h = self.out_proj(torch.cat([fwd, bwd], dim=-1))  # (B, T, d_model)
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

        # GRU-style update gate: controls per-slot retention vs. update
        # gate = sigmoid(W · [prev_memory, candidate]) ∈ (0,1)^(B,N,d)
        self.gate_proj  = nn.Linear(d_model * 2, d_model, bias=True)
        nn.init.zeros_(self.gate_proj.bias)   # start near 0.5 gate (sigmoid(0)=0.5)

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
        N = hints.shape[1]

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

        # Mamba candidate for new memory slots
        candidate = x[:, -N:, :]                                             # (B, N, d)

        if prev_memory is not None:
            # GRU-style update gate: blend Mamba proposal with prev_memory
            # gate → 1: take candidate (update); gate → 0: keep prev_memory (retain)
            gate = torch.sigmoid(
                self.gate_proj(torch.cat([prev_memory, candidate], dim=-1))  # (B, N, d)
            )
            new_memory = self.out_norm(gate * candidate + (1.0 - gate) * prev_memory)
        else:
            new_memory = self.out_norm(candidate)

        return new_memory, enriched_memory



class LocalContextCompressor(nn.Module):
    """
    Returns the last T//ratio frames of the previous clip's visual tokens.
    Provides the most recent local context — the tail of the previous clip
    directly bridges into the current clip temporally.

    (B, T, d_model) → (B, T//ratio, d_model)

    Args:
        ratio:  keep last 1/ratio frames (default 4 → last T//4 frames)
    """

    def __init__(self, ratio: int = 4):
        super().__init__()
        self.ratio = ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T//ratio, d_model)"""
        T = x.shape[1]
        keep = max(T // self.ratio, 1)
        return x[:, -keep:, :]


class TemporalRefiner(nn.Module):
    """
    Stack of N BidirectionalMambaBlocks for temporal refinement of visual tokens.

    Each block runs forward and backward Mamba scans over the full clip,
    giving each frame access to all past and future context within the clip.

    Input:  (B, T, d_model)  — raw VMamba per-frame features
    Output: (B, T, d_model)  — temporally-refined features

    Args:
        d_model:    feature dimension
        num_layers: number of stacked BidirectionalMambaBlocks
    """

    def __init__(self, d_model: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            BidirectionalMambaBlock(d_model)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

"""
Clip-level hint token encoder.

ClipHintEncoder (Q-Former style, BLIP-2)
  Compresses a clip's visual features into N_hints summary tokens via
  iterative query refinement — the same mechanism used in BLIP-2 for
  visual-language bridging.

  Each QFormerBlock applies:
    1. Self-attention among queries  (queries interact with each other)
    2. Cross-attention: Q(queries) × KV(visual features)  (queries read from frames)
    3. FFN refinement

  Stacking N_blocks rounds of this allows queries to progressively refine
  their representation: early blocks broadly locate relevant frames,
  later blocks specialize and extract finer-grained information.

  Learnable hint_queries serve as the initial query state — each query
  learns what type of information to seek (phase-discriminative patterns,
  tool activity, etc.) across all clips.
"""

import torch
import torch.nn as nn


class QFormerBlock(nn.Module):
    """
    Single Q-Former block: self-attn(Q) → cross-attn(Q, V) → FFN.

    Args:
        d_llm:    query dimension
        d_visual: key/value dimension (visual feature dim)
        n_heads:  number of attention heads
        dropout:  dropout rate
    """

    def __init__(self, d_llm: int, d_visual: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # Queries attend to each other
        self.self_attn = nn.MultiheadAttention(
            d_llm, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_sa = nn.LayerNorm(d_llm)

        # Queries attend to visual features (K/V in d_visual space)
        self.cross_attn = nn.MultiheadAttention(
            d_llm, n_heads, dropout=dropout, batch_first=True,
            kdim=d_visual, vdim=d_visual,
        )
        self.norm_ca = nn.LayerNorm(d_llm)

        # FFN
        self.ff = nn.Sequential(
            nn.Linear(d_llm, d_llm * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_llm * 2, d_llm),
        )
        self.norm_ff = nn.LayerNorm(d_llm)

    def forward(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        q: (B, N_hints, d_llm)
        v: (B, T,       d_visual)
        """
        sa, _ = self.self_attn(q, q, q)
        q = self.norm_sa(q + sa)

        ca, _ = self.cross_attn(q, v, v)
        q = self.norm_ca(q + ca)

        q = self.norm_ff(q + self.ff(q))
        return q


class ClipHintEncoder(nn.Module):
    """
    Args:
        n_hints:   number of hint tokens produced per clip
        d_visual:  VMamba output dim (768)
        d_llm:     LLM hidden dim
        n_heads:   attention heads
        dropout:   attention + FFN dropout
        n_blocks:  number of Q-Former blocks (iterative refinement rounds)
        max_seq:   unused, kept for API compatibility
    """

    def __init__(
        self,
        n_hints: int,
        d_visual: int,
        d_llm: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq: int = 512,
        n_blocks: int = 2,
    ):
        super().__init__()
        self.n_hints = n_hints

        # Learnable query tokens — each learns what to look for across all clips
        self.hint_queries = nn.Parameter(torch.randn(1, n_hints, d_llm) * 0.02)

        # Iterative Q-Former blocks
        self.blocks = nn.ModuleList([
            QFormerBlock(d_llm, d_visual, n_heads, dropout)
            for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(d_llm)

    def forward(self, visual_feats: torch.Tensor):
        """
        Args:
            visual_feats: (B, T, d_visual)  temporally-refined VMamba features
        Returns:
            hints:           (B, N_hints, d_llm)
            attn_focus_loss: scalar 0.0  (kept for API compatibility)
        """
        B = visual_feats.shape[0]

        # Expand learnable queries over batch
        q = self.hint_queries.expand(B, -1, -1)    # (B, N_hints, d_llm)

        # Iterative refinement: each block refines q using visual features
        for block in self.blocks:
            q = block(q, visual_feats)              # (B, N_hints, d_llm)

        return self.final_norm(q), visual_feats.new_tensor(0.0)

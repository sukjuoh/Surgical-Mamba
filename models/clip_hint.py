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

    def forward(
        self, q: torch.Tensor, v: torch.Tensor, return_attn: bool = False
    ):
        """
        q: (B, N_hints, d_llm)
        v: (B, T,       d_visual)
        return_attn: if True, also return cross-attn weights (B, N_hints, T)
        """
        sa, _ = self.self_attn(q, q, q)
        q = self.norm_sa(q + sa)

        ca, attn_weights = self.cross_attn(q, v, v, need_weights=return_attn,
                                           average_attn_weights=True)
        q = self.norm_ca(q + ca)

        q = self.norm_ff(q + self.ff(q))
        if return_attn:
            return q, attn_weights  # attn_weights: (B, N_hints, T)
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
        init_embeddings: torch.Tensor = None,
    ):
        super().__init__()
        self.n_hints = n_hints

        # Learnable query tokens — initialized from phase/tool text embeddings if provided,
        # otherwise small random normal (standard init).
        if init_embeddings is not None:
            # init_embeddings: (n_hints, d_llm) — LLM embeddings of surgical concepts.
            # Gives each query a meaningful starting direction (phase/tool semantics)
            # instead of an arbitrary random point in embedding space.
            self.hint_queries = nn.Parameter(init_embeddings.unsqueeze(0).clone().float())
        else:
            self.hint_queries = nn.Parameter(torch.randn(1, n_hints, d_llm) * 0.02)

        # Iterative Q-Former blocks
        self.blocks = nn.ModuleList([
            QFormerBlock(d_llm, d_visual, n_heads, dropout)
            for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(d_llm)
        
    def _attention_focus_loss(self, attn_weights: torch.Tensor, T: int) -> torch.Tensor:
        """
        Attention focus loss: penalises hint i for not concentrating its attention
        on its assigned temporal segment.

        For hint i assigned to segment [start_i, end_i]:
            in_seg = sum of attn_weights[i, start_i:end_i]  ∈ [0, 1]
            loss   = 1 - in_seg   (0 = perfect focus, 1 = no attention in segment)

        Unlike contrastive losses, this works regardless of phase feature distinctiveness —
        it purely measures "did hint i look at the right frames?"

        Args:
            attn_weights: (B, N_hints, T)  attention probabilities from cross_attn
            T:            clip length
        Returns:
            scalar loss (mean over hints and batch)
        """
        N = self.n_hints
        seg_size = max(T // N, 1)
        total = attn_weights.new_tensor(0.0)
        for i in range(N):
            start  = i * seg_size
            end    = start + seg_size if i < N - 1 else T
            in_seg = attn_weights[:, i, start:end].sum(dim=-1)  # (B,)
            total  = total + (1.0 - in_seg).mean()
        return total / N

    def forward(self, visual_feats: torch.Tensor):
        """
        Args:
            visual_feats: (B, T, d_visual)  temporally-refined VMamba features
        Returns:
            hints:           (B, N_hints, d_llm)
            attn_focus_loss: scalar — attention focus loss from last block's cross-attn
        """
        B, T, _ = visual_feats.shape

        # Expand learnable queries over batch
        q = self.hint_queries.expand(B, -1, -1)    # (B, N_hints, d_llm)

        # All blocks except last: standard forward
        for block in self.blocks[:-1]:
            q = block(q, visual_feats)

        # Last block: capture cross-attention weights for focus loss
        q, attn_weights = self.blocks[-1](q, visual_feats, return_attn=True)
        # attn_weights: (B, N_hints, T)

        attn_focus_loss = self._attention_focus_loss(attn_weights, T)

        return self.final_norm(q), attn_focus_loss

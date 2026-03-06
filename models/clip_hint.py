"""
Clip-level hint token encoder.

ClipHintEncoder (improved)
  Compresses a clip's temporally-refined visual features into N_hints tokens.

  Design:
  1. Temporal-segment bias on cross-attention
     Each of the N_hints queries receives a soft Gaussian bias toward its
     assigned temporal segment. The bias is additive on attention logits
     (log-space), so it does not hard-block other frames — just nudges
     each query to specialise on a different part of the clip.
     → Prevents all hints from capturing the same dominant scene.

  2. Transition stream (phase-change detector)
     Frame differences (visual_feats[t] - visual_feats[t-1]) are computed
     and projected as additional K/V alongside the raw features. This gives
     the hints direct access to "what changed between consecutive frames",
     which is the primary signal for detecting phase transitions.

  3. Temporal PE on K/V
     Learnable positional encoding added to visual feats before projection,
     so the hints know where in the clip each frame came from.

  4. Slot identity residual
     Each hint slot has its own learnable embedding (role identity) added
     after cross-attention. This acts as a persistent placeholder so the
     LLM always sees a stable positional/role identity at each hint slot.

  5. Self-attention + FFN
     Hints attend to each other (diversity, redundancy suppression) followed
     by a standard FFN refinement step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipHintEncoder(nn.Module):
    """
    Args:
        n_hints:   number of hint tokens produced per clip
        d_visual:  VMamba output dim (768)
        d_llm:     LLM hidden dim
        n_heads:   attention heads (cross- and self-attention)
        dropout:   attention + FFN dropout
        max_seq:   maximum clip length for temporal PE
    """

    def __init__(
        self,
        n_hints: int,
        d_visual: int,
        d_llm: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq: int = 512,
    ):
        super().__init__()
        self.n_hints = n_hints
        self.d_llm = d_llm

        # ── Temporal PE on visual K/V ────────────────────────────────────────
        # Scale matches standard embedding init: std ≈ 1/√d so PE magnitude
        # is comparable to feature vectors (which are also O(1/√d) after LN).
        self.temporal_pe = nn.Parameter(torch.randn(1, max_seq, d_visual) * (d_visual ** -0.5))

        # ── Main cross-attention stream ──────────────────────────────────────
        self.hint_queries = nn.Parameter(torch.randn(1, n_hints, d_llm) * 0.02)
        self.kv_proj      = nn.Linear(d_visual, d_llm)
        self.cross_attn   = nn.MultiheadAttention(d_llm, n_heads, dropout=dropout, batch_first=True)

        # ── Transition stream (frame-diff cross-attention) ────────────────────
        # Queries from hint_queries attend to frame-difference features.
        # This gives hints a dedicated channel to detect phase transitions.
        self.diff_proj    = nn.Linear(d_visual, d_llm)
        self.diff_attn    = nn.MultiheadAttention(d_llm, n_heads, dropout=dropout, batch_first=True)
        # Gate to blend main + transition streams
        self.stream_gate  = nn.Linear(d_llm * 2, d_llm)

        self.norm_cross   = nn.LayerNorm(d_llm)

        # ── Per-slot identity residual (placeholder role embeddings) ─────────
        self.slot_embeddings = nn.Parameter(torch.randn(1, n_hints, d_llm) * 0.02)

        # ── Self-attention among hint tokens ─────────────────────────────────
        self.self_attn  = nn.MultiheadAttention(d_llm, n_heads, dropout=dropout, batch_first=True)
        self.norm_self  = nn.LayerNorm(d_llm)

        # ── Feed-forward refinement ──────────────────────────────────────────
        self.ff = nn.Sequential(
            nn.Linear(d_llm, d_llm * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_llm * 2, d_llm),
        )
        self.norm_ff = nn.LayerNorm(d_llm)

    def _segment_bias(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Returns (n_hints, T) additive bias for attention logits.
        Hint i is softly biased toward its temporal segment.
        sigma = segment_width / 2  (controls how tightly each hint focuses).
        """
        # Segment centers: evenly spaced in [0, T-1]
        centers = torch.linspace(0, T - 1, self.n_hints, device=device)   # (n_hints,)
        positions = torch.arange(T, dtype=torch.float32, device=device)    # (T,)
        dist2 = (centers.unsqueeze(1) - positions.unsqueeze(0)) ** 2       # (n_hints, T)
        sigma2 = max((T / self.n_hints / 2) ** 2, 1.0)
        return -dist2 / (2 * sigma2)  # (n_hints, T)  — log-scale (negative = suppress far frames)

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
            hints:          (B, N_hints, d_llm)
            attn_focus_loss: scalar — attention focus loss (supervised segment attention)
        """
        B, T, _ = visual_feats.shape
        device = visual_feats.device

        # 1. Temporal PE on visual features ───────────────────────────────────
        feats_pe = visual_feats + self.temporal_pe[:, :T, :]           # (B, T, d_visual)

        # 2. Main cross-attention stream ──────────────────────────────────────
        kv_main = self.kv_proj(feats_pe)                               # (B, T, d_llm)
        Q = self.hint_queries.expand(B, -1, -1)                        # (B, N_hints, d_llm)

        # Temporal-segment bias: (n_hints, T) additive on attention logits
        seg_bias = self._segment_bias(T, device)                       # (N_hints, T)
        # MHA attn_mask must be 2D (N_hints, T) for broadcast over B and heads,
        # or 3D (B*n_heads, N_hints, T). Use 2D so batch size doesn't matter.
        main_out, attn_weights = self.cross_attn(
            Q, kv_main, kv_main,
            attn_mask=seg_bias,                                        # (N_hints, T)
            need_weights=True,
            average_attn_weights=True,
        )                                                # main_out: (B, N_hints, d_llm)
                                                         # attn_weights: (B, N_hints, T)

        # 3. Transition stream ────────────────────────────────────────────────
        # Frame differences: detect where things change
        diff = feats_pe[:, 1:, :] - feats_pe[:, :-1, :]               # (B, T-1, d_visual)
        diff = F.pad(diff, (0, 0, 1, 0))                               # (B, T, d_visual) pad t=0
        kv_diff = self.diff_proj(diff)                                 # (B, T, d_llm)
        diff_out, _ = self.diff_attn(Q, kv_diff, kv_diff)             # (B, N_hints, d_llm)

        # 4. Gated fusion of main + transition streams ────────────────────────
        gate = torch.sigmoid(self.stream_gate(
            torch.cat([main_out, diff_out], dim=-1)                    # (B, N_hints, 2*d_llm)
        ))                                                             # (B, N_hints, d_llm)
        hints = gate * main_out + (1 - gate) * diff_out               # (B, N_hints, d_llm)

        # Add slot identity residual → stable placeholder per position, then LN
        hints = self.norm_cross(hints + self.slot_embeddings)          # (B, N_hints, d_llm)

        # 5. Self-attention among hints (diversity + interaction) ──────────────
        sa, _ = self.self_attn(hints, hints, hints)
        hints = self.norm_self(hints + sa)

        # 6. FFN refinement ───────────────────────────────────────────────────
        hints = self.norm_ff(hints + self.ff(hints))

        # 7. Attention focus loss ──────────────────────────────────────────────
        # Uses attn_weights from the main cross-attention stream.
        # Measures how well each hint concentrated on its assigned segment.
        attn_focus_loss = self._attention_focus_loss(attn_weights, T)

        return hints, attn_focus_loss

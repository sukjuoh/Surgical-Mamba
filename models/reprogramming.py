"""
Reprogramming layer from Time-LLM (Jin et al., ICLR 2024).
Cross-attention between visual patch embeddings (Q) and LLM word embeddings (K/V),
mapping visual tokens into the LLM's embedding space.

Linear attention variant (denominator-normalized):
  Uses φ(x) = ELU(x) + 1 as the kernel feature map, then computes:

    out[q] = φ(Q[q]) · (Σ_k φ(K[k]) ⊗ V[k]) / (φ(Q[q]) · Σ_k φ(K[k]))

  The denominator guarantees each query's output is a normalized convex combination
  of values — preserving the softmax invariant while reducing O(T×S) → O((T+S)×d_k).

  Since word_embeddings (K/V) are fixed within a forward pass, the KV context
  matrix (Σ_k φ(K[k]) ⊗ V[k]) can be computed once per call and shared across
  all T query tokens, which is where the actual speedup comes from.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """φ(x) = ELU(x) + 1  — non-negative, stable kernel feature map."""
    return F.elu(x) + 1


class ReprogrammingLayer(nn.Module):
    """
    Maps visual embeddings (d_model) into LLM embedding space (d_llm) via
    normalized linear cross-attention where K/V come from a reduced set of
    LLM word embeddings.

    Args:
        d_model:           dimension of input visual embeddings (e.g. 768 from VMamba)
        n_heads:           number of attention heads
        d_llm:             LLM hidden dimension
        d_keys:            per-head key/query dimension (default: d_model // n_heads)
        attention_dropout: kept for API compatibility; applied to output projection
    """

    def __init__(self, d_model: int, n_heads: int, d_llm: int, d_keys: int = None, attention_dropout: float = 0.1):
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_proj  = nn.Linear(d_model, d_keys * n_heads)
        self.key_proj    = nn.Linear(d_llm,   d_keys * n_heads)
        self.value_proj  = nn.Linear(d_llm,   d_keys * n_heads)
        self.out_proj    = nn.Linear(d_keys * n_heads, d_llm)

        self.n_heads = n_heads
        self.d_keys  = d_keys
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, x: torch.Tensor, word_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:               (B, T, d_model)  visual frame tokens
            word_embeddings: (S, d_llm)        reduced LLM vocabulary embeddings
        Returns:
            (B, T, d_llm)   reprogrammed tokens aligned to LLM space
        """
        B, T, _ = x.shape
        S = word_embeddings.shape[0]
        H = self.n_heads
        d_k = self.d_keys

        # Project to multi-head format
        Q = self.query_proj(x).view(B, T, H, d_k)             # (B, T, H, d_k)
        K = self.key_proj(word_embeddings).view(S, H, d_k)    # (S, H, d_k)
        V = self.value_proj(word_embeddings).view(S, H, d_k)  # (S, H, d_k)

        # Apply ELU+1 kernel map — keeps all values non-negative
        Q = _elu_feature_map(Q)   # (B, T, H, d_k)
        K = _elu_feature_map(K)   # (S, H, d_k)

        # ── Linear attention in O((T+S) × d_k) ───────────────────────────────
        # KV context: Σ_s φ(K[s]) ⊗ V[s]  →  (H, d_k, d_k)
        # Computed once and shared across all T queries.
        kv = torch.einsum("shd,she->hde", K, V)   # (H, d_k, d_k)

        # Numerator: φ(Q) · KV  →  (B, T, H, d_k)
        num = torch.einsum("bthd,hde->bthe", Q, kv)

        # Denominator: φ(Q) · Σ_s φ(K[s])  →  (B, T, H, 1)
        # Σ_s φ(K[s]): (H, d_k)
        k_sum = K.sum(dim=0)                                   # (H, d_k)
        denom = torch.einsum("bthd,hd->bth", Q, k_sum).unsqueeze(-1).clamp(min=1e-6)

        # Normalized output: (B, T, H, d_k)
        out = num / denom
        out = self.dropout(out)
        out = out.reshape(B, T, -1)    # (B, T, H*d_k)

        return self.out_proj(out)      # (B, T, d_llm)

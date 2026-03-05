"""
Reprogramming layer from Time-LLM (Jin et al., ICLR 2024).
Cross-attention between visual patch embeddings (Q) and LLM word embeddings (K/V),
mapping visual tokens into the LLM's embedding space.
"""

from math import sqrt
import torch
import torch.nn as nn


class ReprogrammingLayer(nn.Module):
    """
    Maps visual embeddings (d_model) into LLM embedding space (d_llm) via
    multi-head cross-attention where K/V come from a reduced set of LLM word embeddings.

    Args:
        d_model: dimension of input visual embeddings (e.g. 768 from VMamba)
        n_heads: number of attention heads
        d_llm: LLM hidden dimension (e.g. 768 for GPT-2, 4096 for LLaMA-7B)
        d_keys: per-head key/query dimension (default: d_model // n_heads)
        attention_dropout: dropout on attention weights
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
        self.scale   = sqrt(d_keys)
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

        # Project to multi-head format
        Q = self.query_proj(x).view(B, T, H, -1)              # (B, T, H, d_k)
        K = self.key_proj(word_embeddings).view(S, H, -1)     # (S, H, d_k)
        V = self.value_proj(word_embeddings).view(S, H, -1)   # (S, H, d_k)

        # Cross-attention: each frame token attends over all word embeddings
        # scores: (B, H, T, S)
        scores = torch.einsum("bthd,shd->bhts", Q, K) / self.scale
        attn   = self.dropout(torch.softmax(scores, dim=-1))  # (B, H, T, S)

        # Aggregate values: (B, T, H, d_k)
        out = torch.einsum("bhts,shd->bthd", attn, V)
        out = out.reshape(B, T, -1)   # (B, T, H*d_k)

        return self.out_proj(out)     # (B, T, d_llm)

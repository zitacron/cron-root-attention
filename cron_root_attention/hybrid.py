"""
Cron Root Attention - Hybrid Backend Selection
===============================================

Automatically selects between SDPA and Cron Root based on sequence length.

(c) 2026 Zitacron. All rights reserved.
Licensed under Apache 2.0 - See LICENSE file for details.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

from .core import cron_root_attention_v14

# Training crossover: Cron Root beats SDPA at S >= ~1536 (training),
# S >= ~1024 (forward-only), S >= ~1024 (inference).
# Default to 1536 as the training-oriented threshold (conservative).
_SDPA_THRESHOLD = 1536


def cron_root_attention_hybrid(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    backend: Literal["auto", "cron_root", "sdpa"] = "auto",
    threshold: int = _SDPA_THRESHOLD,
) -> torch.Tensor:
    """
    Hybrid Cron Root attention with automatic backend selection.
    
    Uses SDPA/FlashAttention for S < threshold, Cron Root for S >= threshold.
    In "auto" mode, selects the faster backend based on measured crossover:
      - Training crossover: ~1536 tokens
      - Forward/inference crossover: ~1024 tokens
    
    Args:
        q: Query tensor (B, H, S, D)
        k: Key tensor (B, H, S, D)
        v: Value tensor (B, H, S, D)
        backend: "auto" (based on S), "cron_root" (force √N), "sdpa" (force SDPA)
        threshold: Sequence length threshold for auto selection (default: 1536)
        
    Returns:
        Output tensor (B, H, S, D)
        
    Example:
        >>> output = cron_root_attention_hybrid(q, k, v, backend="auto")
        >>> # Uses SDPA for S < 1536, Cron Root for S >= 1536
    """
    S = q.shape[2]
    
    if backend == "cron_root":
        return cron_root_attention_v14(q, k, v)
    elif backend == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:  # auto
        if S >= threshold:
            return cron_root_attention_v14(q, k, v)
        else:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class CronRootAttentionHybrid(nn.Module):
    """
    Hybrid attention module with automatic backend selection.
    
    Uses SDPA for short sequences and Cron Root for long sequences.
    
    Args:
        d_model: Total dimension of the model
        n_heads: Number of attention heads
        threshold: Sequence length threshold for Cron Root (default: 1024)
        
    Example:
        >>> attn = CronRootAttentionHybrid(d_model=1024, n_heads=16)
        >>> x_short = torch.randn(1, 512, 1024)   # Uses SDPA
        >>> x_long = torch.randn(1, 8192, 1024)   # Uses Cron Root
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        threshold: int = _SDPA_THRESHOLD,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.threshold = threshold
        self.dropout = dropout
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        
        # Project
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Hybrid attention
        out = cron_root_attention_hybrid(q, k, v, backend="auto", threshold=self.threshold)
        
        # Reshape and output projection
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out)
        
        if self.dropout > 0 and self.training:
            out = F.dropout(out, p=self.dropout)
        
        return out

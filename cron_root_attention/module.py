"""
Cron Root Attention - Module API
=================================

Drop-in replacement modules for PyTorch's MultiheadAttention.

(c) 2026 Zitacron. All rights reserved.
Licensed under Apache 2.0 - See LICENSE file for details.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .core import cron_root_attention_v14


class CronRootAttention(nn.Module):
    """
    Cron Root √N Sparse Attention layer with O(N√N) complexity.
    
    Replaces standard scaled dot-product attention with a sparse pattern
    that attends to local window (√N) + strided positions (√N).
    
    Args:
        d_model: Total dimension of the model
        n_heads: Number of attention heads
        n_kv_heads: Number of key-value heads (for GQA). Defaults to n_heads.
        dropout: Dropout probability (applied after attention)
        bias: Whether to include bias in projections
        
    Example:
        >>> attn = CronRootAttention(d_model=1024, n_heads=16)
        >>> x = torch.randn(1, 8192, 1024, device='cuda', dtype=torch.float16)
        >>> output = attn(x)  # 14x faster than standard attention at S=8192
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.kv_groups = n_heads // self.n_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.o_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with √N sparse attention.
        
        Args:
            x: Query input (B, S, D)
            kv: Key-value input (B, S, D). If None, uses x (self-attention).
            
        Returns:
            Output tensor (B, S, D)
        """
        B, S, D = x.shape
        kv = kv if kv is not None else x
        
        # Project to Q, K, V
        q = self.q_proj(x)     # (B, S, D)
        k = self.k_proj(kv)    # (B, S, D_kv)
        v = self.v_proj(kv)    # (B, S, D_kv)
        
        # Reshape for multi-head attention
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D_h)
        k = k.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Expand KV for GQA if needed
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)
        
        # Apply √N attention
        out = cron_root_attention_v14(q, k, v)  # (B, H, S, D_h)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, S, D)  # (B, S, D)
        
        # Output projection
        out = self.o_proj(out)
        
        if self.dropout > 0 and self.training:
            out = F.dropout(out, p=self.dropout)
        
        return out


class CronRootMultiheadAttention(nn.Module):
    """
    Cron Root Multi-Head Attention - Drop-in replacement for nn.MultiheadAttention.
    
    Uses √N sparse attention pattern for O(N√N) complexity instead of O(N²).
    
    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Add bias to projections
        batch_first: If True, input is (B, S, D), else (S, B, D)
        
    Example:
        >>> mha = CronRootMultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        >>> x = torch.randn(2, 1024, 512, device='cuda', dtype=torch.float16)
        >>> output, _ = mha(x, x, x)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Combined QKV projection (like nn.MultiheadAttention)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass compatible with nn.MultiheadAttention.
        
        Note: √N attention is inherently causal. key_padding_mask and attn_mask
        are currently ignored (TODO: implement padding mask support).
        
        Args:
            query, key, value: Input tensors (S, B, D) or (B, S, D) if batch_first
            key_padding_mask: Ignored (not yet supported)
            need_weights: If True, returns attention weights (as None for sparse)
            attn_mask: Ignored (√N pattern is used instead)
            average_attn_weights: Ignored
            is_causal: Ignored (always causal)
            
        Returns:
            Tuple of (output, None)  # Attention weights not returned for sparse
        """
        # Handle batch_first
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        B, S, D = query.shape
        
        # Self-attention only for now
        assert query is key and key is value, "Cross-attention not yet supported"
        
        # Combined QKV projection
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, D_h)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply √N attention
        out = cron_root_attention_v14(q, k, v)  # (B, H, S, D_h)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        
        # Output projection
        out = self.out_proj(out)
        
        if self.dropout > 0 and self.training:
            out = F.dropout(out, p=self.dropout)
        
        # Handle batch_first for output
        if not self.batch_first:
            out = out.transpose(0, 1)
        
        # Return None for attention weights (sparse pattern doesn't have dense weights)
        return out, None

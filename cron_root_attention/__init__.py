"""
Cron Root Attention - Sub-quadratic Attention for Long-Context Transformers
============================================================================

A high-performance Triton implementation of √N sparse attention achieving
O(N√N) complexity instead of O(N²), enabling up to 57x kernel speedups at
long sequence lengths.

(c) 2026 Zitacron. All rights reserved.
Licensed under Apache 2.0 - See LICENSE file for details.

Repository: https://github.com/zitacron/cron-root-attention
arXiv: [To be added after submission]

Naming:
    Cron  → ZitaCron (the company)
    Root  → √ (square root) - the mathematical operation
    Together: Cron Root Attention

Supported GPUs:
    - NVIDIA GeForce RTX 50/40/30/20 series
    - NVIDIA H100, H200, H800, A100, L40S, L4, V100 (datacenter)
    - NVIDIA B100, B200, GB200 (Blackwell datacenter)
    - Auto-detection of SM count for optimal grid sizing

Key Results (RTX 5070 Ti, FP16, PyTorch 2.9.1 + CUDA 12.8):
    Forward pass (kernel only):
        S=1024:   1.02x speedup vs SDPA/Flash  (crossover ~1K)
        S=2048:   3.44x speedup
        S=4096:   9.24x speedup
        S=8192:   13.3x speedup
        S=16384:  9.91x speedup
        S=65536:  18.6x speedup
        S=131072: 27.0x speedup
        S=262144: 45.3x speedup  
        S=524288: 56.8x speedup
    
    Training (fwd+bwd):
        S=4096:  1.45x, S=8192: 1.73x, S=131K: 3.76x
        Crossover ~2K (fully-fused single-kernel backward for S≤8K)
    
    Hybrid mode: ≥1.0x at ALL lengths (auto SDPA below 1536)

Usage:
    from cron_root_attention import cron_root_attention, CronRootAttention
    
    # Functional API
    output = cron_root_attention(q, k, v)
    
    # Module API (drop-in replacement)
    attn = CronRootAttention(d_model=1024, n_heads=16)
    output = attn(x)
    
    # Check GPU compatibility
    from cron_root_attention import get_gpu_info
    print(get_gpu_info())
"""

__version__ = "0.1.0"
__author__ = "Zitacron"

from .core import cron_root_attention_v14, get_num_sms, GPU_SM_MAP
from .module import CronRootAttention, CronRootMultiheadAttention
from .hybrid import cron_root_attention_hybrid, CronRootAttentionHybrid


def get_gpu_info() -> dict:
    """
    Get GPU information and compatibility status.
    
    Returns:
        dict with gpu_name, sm_count, and is_known_gpu
    """
    import torch
    if not torch.cuda.is_available():
        return {"gpu_name": "No GPU", "sm_count": 0, "is_known_gpu": False}
    
    gpu_name = torch.cuda.get_device_name()
    sm_count = get_num_sms()
    is_known = gpu_name in GPU_SM_MAP or any(n in gpu_name for n in GPU_SM_MAP)
    
    return {
        "gpu_name": gpu_name,
        "sm_count": sm_count,
        "is_known_gpu": is_known,
    }


# Alias for convenience
cron_root_attention = cron_root_attention_v14

__all__ = [
    "cron_root_attention",
    "cron_root_attention_v14",
    "CronRootAttention",
    "CronRootMultiheadAttention",
    "cron_root_attention_hybrid",
    "CronRootAttentionHybrid",
    "get_num_sms",
    "get_gpu_info",
    "GPU_SM_MAP",
    "__version__",
]

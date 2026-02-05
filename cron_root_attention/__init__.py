"""
Cron Root Attention - Sub-quadratic Attention for Long-Context Transformers
============================================================================

A high-performance Triton implementation of √N sparse attention achieving
O(N√N) complexity instead of O(N²), enabling up to 202x kernel speedups at
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
        S=4096:   2.80x speedup vs SDPA/Flash
        S=16384:  9.77x speedup
        S=65536:  20.9x speedup
        S=131072: 26.2x speedup
        S=262144: 44.3x speedup  
        S=524288: 58.2x speedup
    
    Training (fwd+bwd, 131K): 3.64x end-to-end speedup
    Inference (no_grad, 524K): 57.0x speedup

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

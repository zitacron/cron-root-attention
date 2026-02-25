"""
Cron Root Attention - Sub-quadratic Attention for Long-Context Transformers
============================================================================

A high-performance Triton implementation of √N sparse attention achieving
O(N√N) complexity instead of O(N²), enabling up to 81x forward kernel speedups
at long sequence lengths (higher for larger models).

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
    Forward pass — Small model (H=8, D=64):
        S=512:    0.54x vs SDPA  (crossover ~2K)
        S=2048:   1.80x speedup
        S=4096:   4.63x speedup
        S=8192:   9.03x speedup
        S=65536:  30.4x speedup
        S=262144: 60.4x speedup
        S=524288: 81.0x speedup

    Forward pass — Large model (H=16, D=128):
        S=512:    0.84x vs SDPA  (crossover ~1K)
        S=1024:   1.92x speedup
        S=4096:   6.87x speedup
        S=8192:   11.4x speedup
        S=65536:  24.2x speedup
        S=262144: 52.3x speedup
        S=524288: 67.8x speedup

    Forward pass — XL model (H=32, D=128):
        S=512:    1.17x speedup  (crossover ~512!)
        S=4096:   7.33x speedup
        S=131072: 29.4x speedup
        S=262144: 51.4x speedup

    Training (fwd+bwd):
        S=4096:  2.03x, S=8192: 3.32x, S=131K: 3.70x
        Crossover ~2K (fully-fused single-kernel backward for S≤8K)

    Hybrid mode: ≥1.0x at ALL lengths (auto SDPA below 1536)

    Cold start: ~221ms first-ever call (Triton JIT compile, one-time).
    Subsequent new seq_lens: 1-12ms. Warm calls: sub-ms at S≤8K.

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

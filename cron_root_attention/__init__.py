"""
Cron Root Attention - Sub-quadratic Attention for Long-Context Transformers
============================================================================

A high-performance Triton implementation of √N sparse attention achieving
O(N√N) complexity instead of O(N²), enabling up to 79x forward kernel speedups
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
        S=512:    0.53x vs SDPA  (crossover ~2K)
        S=2048:   1.95x speedup
        S=4096:   4.27x speedup
        S=8192:   8.90x speedup
        S=65536:  27.7x speedup
        S=262144: 62.3x speedup
        S=524288: 78.6x speedup

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
        S=4096:  1.83x, S=8192: 3.07x, S=131K: 3.64x
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

from .core import cron_root_attention_v14, get_num_sms, GPU_SM_MAP, _tuner
from .module import CronRootAttention, CronRootMultiheadAttention
from .hybrid import cron_root_attention_hybrid, CronRootAttentionHybrid


def calibrate(H: int = 8, D: int = 64, dtype=None,
              S: int = 4096, verbose: bool = True) -> dict:
    """Optional pre-warm: benchmark kernel configs for (H, D) before training.

    Not required — the auto-tuner runs transparently on the first real forward
    and backward calls.  Call this at model init for deterministic startup
    (avoids a ~10ms inline benchmark on the first training step).

    The optimal (num_warps, num_stages) depends on (GPU, H, D) only — it is
    S-independent (verified empirically across S=512..131072).  A single test
    at ``S`` (default 4096, always fits in VRAM) characterizes the GPU.

    Args:
        H:      Number of attention heads.
        D:      Head dimension.
        dtype:  Tensor dtype (default: torch.float16).
        S:      Sequence length for the calibration run (default: 4096).
        verbose:  Print results.

    Returns:
        dict with keys ``'fwd'`` and ``'bwd'``, each mapping to ``(num_warps, num_stages)``.
    """
    import torch, gc, math
    if dtype is None:
        dtype = torch.float16

    # Check cache first
    fwd_done = _tuner.is_tuned("fwd", H, D)
    bwd_done = _tuner.is_tuned("bwd", H, D)
    if fwd_done and bwd_done:
        fwd_cfg = _tuner.get_config("fwd", H, D)
        bwd_cfg = _tuner.get_config("bwd", H, D)
        if verbose:
            print(f"  H={H} D={D}: fwd=w{fwd_cfg[0]}s{fwd_cfg[1]}  "
                  f"bwd=w{bwd_cfg[0]}s{bwd_cfg[1]}  (cached)")
        return {"fwd": fwd_cfg, "bwd": bwd_cfg}

    dev = torch.device("cuda", torch.cuda.current_device())
    B = 1
    gc.collect()
    torch.cuda.empty_cache()

    SQRT_N = int(math.ceil(math.sqrt(S)))
    NUM_RELAY = (S + SQRT_N - 1) // SQRT_N
    BLOCK_M = BLOCK_D = BLOCK_LOCAL = BLOCK_STRIDE = BLOCK_RELAY = 64

    q = torch.randn(B, H, S, D, device=dev, dtype=dtype)
    k = torch.randn(B, H, S, D, device=dev, dtype=dtype)
    v = torch.randn(B, H, S, D, device=dev, dtype=dtype)
    o = torch.empty_like(q)
    L = torch.empty(B, H, S, dtype=torch.float32, device=dev)
    relay_k = torch.randn(B, H, NUM_RELAY, D, device=dev, dtype=dtype)
    relay_v = torch.randn(B, H, NUM_RELAY, D, device=dev, dtype=dtype)

    # ── Forward ───────────────────────────────────────────────────────
    fwd_cfg = _tuner.tune_fwd_inline(
        q, k, v, o, L, relay_k, relay_v,
        S, D, SQRT_N, NUM_RELAY, BLOCK_RELAY,
        BLOCK_M, BLOCK_D, BLOCK_LOCAL, BLOCK_STRIDE,
        kv_groups=1, H_q=H, B=B)

    # ── Backward (via dQ kernel, representative of all bwd kernels) ──
    from .core import _cron_root_attn_bwd_dq_only_v14
    num_tiles_m = (S + BLOCK_M - 1) // BLOCK_M
    dq = torch.empty_like(q)
    do_tensor = torch.randn_like(q)

    bwd_cfg = _tuner.tune_bwd_inline(
        _cron_root_attn_bwd_dq_only_v14,
        (B, H, num_tiles_m),
        (q, k, v, o, do_tensor, dq, L,
         relay_k, relay_v,
         q.stride(0), q.stride(1), q.stride(2), q.stride(3),
         k.stride(0), k.stride(1), k.stride(2), k.stride(3),
         v.stride(0), v.stride(1), v.stride(2), v.stride(3),
         o.stride(0), o.stride(1), o.stride(2), o.stride(3),
         do_tensor.stride(0), do_tensor.stride(1), do_tensor.stride(2), do_tensor.stride(3),
         dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
         L.stride(0), L.stride(1), L.stride(2),
         relay_k.stride(0), relay_k.stride(1), relay_k.stride(2), relay_k.stride(3),
         relay_v.stride(0), relay_v.stride(1), relay_v.stride(2), relay_v.stride(3)),
        dict(S=S, D=D, SQRT_N=SQRT_N,
             NUM_RELAY=0, BLOCK_RELAY=BLOCK_RELAY,
             BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
             BLOCK_LOCAL=BLOCK_LOCAL, BLOCK_STRIDE=BLOCK_STRIDE),
        H=H, D=D)

    del q, k, v, o, L, relay_k, relay_v, dq, do_tensor
    gc.collect()
    torch.cuda.empty_cache()

    if verbose:
        print(f"  H={H} D={D}: fwd=w{fwd_cfg[0]}s{fwd_cfg[1]}  "
              f"bwd=w{bwd_cfg[0]}s{bwd_cfg[1]}  "
              f"(saved to {_tuner._cache_file})")

    return {"fwd": fwd_cfg, "bwd": bwd_cfg}


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
    "calibrate",
    "get_num_sms",
    "get_gpu_info",
    "GPU_SM_MAP",
    "__version__",
]

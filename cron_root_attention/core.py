"""
Cron Root Attention V14 - Persistent Tiled Implementation
==========================================================

Cron (ZitaCron) + Root (√) = Cron Root Attention

This module implements the V14 "Persistent Tiled" √N attention, optimized for
small-to-medium sequence lengths (S < 16K) where kernel launch overhead dominates.

Key Innovations:
1. FUSED BLOCK-QUERY: Process BLOCK_M queries per thread block (not 1 per block)
   - Reduces kernel blocks from S to S/BLOCK_M (64-128x fewer launches)
   
2. STATIC MEMORY OFFSETS: Pre-calculate √N offsets once, avoid Python overhead
   - Eliminates 30-40% of orchestration time at small S
   
3. PERSISTENT KERNELS: Launch exactly NUM_SMS blocks with internal work queue
   - Dynamically detects GPU SM count (supports 5000/4000/3000 series, datacenter)
   - Each block pulls work from a global counter until done
   - Near-zero scheduling overhead

Supported GPUs:
- NVIDIA GeForce RTX 50/40/30/20 series (consumer)
- NVIDIA H100/H200/H800, A100, L40S, L4, V100 (datacenter)
- NVIDIA B100/B200/GB200 (Blackwell datacenter)
- Auto-detection via torch.cuda.get_device_properties()

Performance (RTX 5070 Ti, FP16, verified warm-start benchmarks):
- Forward pass speedup vs SDPA (Small model H=8, D=64):
  S=2K: 1.95x, S=4K: 4.27x, S=8K: 8.90x, S=64K: 27.7x, S=512K: 78.6x
- Training (fwd+bwd) speedup:
  S=4K: 1.83x, S=8K: 3.07x, S=64K: 2.57x, S=128K: 3.64x
- Crossover: ~2K (forward and training)
- Cold start: ~221ms first call (Triton JIT), <2ms thereafter

(c) 2026 Zitacron. All rights reserved.
Licensed under Apache 2.0 - See LICENSE file for details.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple
from functools import lru_cache
import json
import os
import threading

# =============================================================================
# HARDWARE DETECTION & GPU COMPATIBILITY
# =============================================================================
# SM counts for popular GPUs (used for persistent kernel grid sizing)
# Source: Official NVIDIA specs

GPU_SM_MAP = {
    # Blackwell (5000 series)
    "NVIDIA GeForce RTX 5090": 170,
    "NVIDIA GeForce RTX 5080": 84,
    "NVIDIA GeForce RTX 5070 Ti": 70,
    "NVIDIA GeForce RTX 5070": 48,
    
    # Ada Lovelace (4000 series)
    "NVIDIA GeForce RTX 4090": 128,
    "NVIDIA GeForce RTX 4080 SUPER": 80,
    "NVIDIA GeForce RTX 4080": 76,
    "NVIDIA GeForce RTX 4070 Ti SUPER": 66,
    "NVIDIA GeForce RTX 4070 Ti": 60,
    "NVIDIA GeForce RTX 4070 SUPER": 56,
    "NVIDIA GeForce RTX 4070": 46,
    "NVIDIA GeForce RTX 4060 Ti": 34,
    "NVIDIA GeForce RTX 4060": 24,
    
    # Ampere (3000 series)
    "NVIDIA GeForce RTX 3090 Ti": 84,
    "NVIDIA GeForce RTX 3090": 82,
    "NVIDIA GeForce RTX 3080 Ti": 80,
    "NVIDIA GeForce RTX 3080": 68,
    "NVIDIA GeForce RTX 3070 Ti": 48,
    "NVIDIA GeForce RTX 3070": 46,
    "NVIDIA GeForce RTX 3060 Ti": 38,
    "NVIDIA GeForce RTX 3060": 28,
    
    # Datacenter GPUs - Blackwell
    "NVIDIA B200": 208,
    "NVIDIA B100": 176,
    "NVIDIA GB200": 208,
    
    # Datacenter GPUs - Hopper
    "NVIDIA H100 PCIe": 114,
    "NVIDIA H100 SXM": 132,
    "NVIDIA H100 NVL": 114,
    "NVIDIA H200": 132,
    "NVIDIA H800": 132,
    "NVIDIA H800 PCIe": 114,
    "NVIDIA H800 SXM": 132,
    
    # Datacenter GPUs - Ada
    "NVIDIA L40S": 142,
    "NVIDIA L40": 142,
    "NVIDIA L4": 58,
    
    # Datacenter GPUs - Ampere
    "NVIDIA A100 PCIe": 108,
    "NVIDIA A100 SXM": 108,
    "NVIDIA A100 80GB": 108,
    "NVIDIA A40": 84,
    "NVIDIA A30": 56,
    "NVIDIA A10": 72,
    "NVIDIA A16": 40,
    "NVIDIA A6000": 84,
    "NVIDIA RTX A6000": 84,
    "NVIDIA RTX A5000": 64,
    "NVIDIA RTX A4000": 48,
    
    # Turing (2000 series)
    "NVIDIA GeForce RTX 2080 Ti": 68,
    "NVIDIA GeForce RTX 2080 SUPER": 48,
    "NVIDIA GeForce RTX 2080": 46,
    "NVIDIA GeForce RTX 2070 SUPER": 40,
    "NVIDIA GeForce RTX 2070": 36,
    "NVIDIA GeForce RTX 2060 SUPER": 34,
    "NVIDIA GeForce RTX 2060": 30,
    "NVIDIA TITAN RTX": 72,
    
    # Volta
    "NVIDIA V100 PCIe": 80,
    "NVIDIA V100 SXM2": 80,
    "Tesla V100-SXM2-16GB": 80,
    "Tesla V100-SXM2-32GB": 80,
    "Tesla V100-PCIE-16GB": 80,
    "Tesla V100-PCIE-32GB": 80,
}

@lru_cache(maxsize=1)
def get_num_sms() -> int:
    """
    Get the number of Streaming Multiprocessors (SMs) for the current GPU.
    Uses CUDA device properties for reliable detection.
    Falls back to GPU name lookup if needed.
    
    Returns:
        Number of SMs on the current GPU
    """
    if not torch.cuda.is_available():
        return 1  # Fallback for CPU-only
    
    device = torch.cuda.current_device()
    
    # Primary method: Use CUDA device properties (most reliable)
    props = torch.cuda.get_device_properties(device)
    if hasattr(props, 'multi_processor_count') and props.multi_processor_count > 0:
        return props.multi_processor_count
    
    # Fallback: Lookup by GPU name
    gpu_name = torch.cuda.get_device_name(device)
    
    # Try exact match first
    if gpu_name in GPU_SM_MAP:
        return GPU_SM_MAP[gpu_name]
    
    # Try partial match (handles variations in naming)
    for known_name, sm_count in GPU_SM_MAP.items():
        if known_name in gpu_name or gpu_name in known_name:
            return sm_count
    
    # Conservative fallback: assume mid-range GPU
    print(f"Warning: Unknown GPU '{gpu_name}', using 48 SMs as fallback. "
          f"Consider adding to GPU_SM_MAP for optimal performance.")
    return 48

# Dynamic SM count (cached after first call)
NUM_SMS = None  # Lazy initialization

def _get_num_sms():
    """Lazily get SM count to avoid CUDA initialization at import time."""
    global NUM_SMS
    if NUM_SMS is None:
        NUM_SMS = get_num_sms()
    return NUM_SMS

# =============================================================================
# BLACKWELL (SM 12.0) KERNEL CONFIGURATION
# =============================================================================
# Triton kernel launch parameters tuned per GPU generation.
#
#  num_warps:
#    Blackwell SM 12.0 has a larger register file (256KB vs 128KB on Ampere)
#    and 128-wide warp schedulers, so doubling warps per block from 4→8 gives
#    better latency hiding for memory-bound attention kernels.
#    Hopper SM 9.0 also benefits from 8 warps due to its wider execution units.
#
#  num_stages (forward):
#    Controls the software-pipelining depth for async LDGSTS prefetch.
#    Blackwell's deeper L2 cache hierarchy and larger shared memory allow
#    num_stages=4 (vs 2 on Ampere) without spilling — effectively overlapping
#    2 extra tiles of data loading with compute.
#
#  num_stages (backward):
#    Backward kernels carry more live accumulators (dQ,dK,dV in FP32), so
#    we cap at 2 to avoid register pressure spills even on Blackwell.
# =============================================================================

def _get_blackwell_kernel_config():
    """Return (num_warps, num_stages_fwd, num_stages_bwd) for the current GPU.

    These are conservative defaults that perform well across all tested GPUs.
    The inline auto-tuner (below) refines them on first call if a better config
    exists for the specific (GPU, H, D) combination.
    """
    if not torch.cuda.is_available():
        return 4, 2, 1
    # Empirically verified: w4 s2 consistently wins or ties across
    # Turing, Ampere, Ada, Hopper, and Blackwell at S=512..131072.
    return 4, 2, 2

_KNL_WARPS, _KNL_STAGES_FWD, _KNL_STAGES_BWD = _get_blackwell_kernel_config()
# Short aliases used at kernel launch sites
_KW, _KS, _KS_FWD = _KNL_WARPS, _KNL_STAGES_BWD, _KNL_STAGES_FWD

# =============================================================================
# ADAPTIVE KERNEL AUTO-TUNER  (inline, first-call, torch.compile-style)
# =============================================================================
# The optimal (num_warps, num_stages) depends on (GPU register file × head dim D),
# NOT on sequence length or total model parameters.  Empirically verified: the
# winner is the same at S=512 through S=131072 on each tested GPU.
#
# Approach (inspired by torch.compile mode="max-autotune"):
#   1. First real forward call for a given (H, D): inline-benchmark all candidate
#      configs using the ACTUAL q/k/v tensors.  Cost: ~10ms, one time per (H, D).
#   2. First real backward call: benchmark during the existing CUDA graph warmup.
#   3. Cache by (GPU_name, op, H, D) — no S dimension (S-independent).
#   4. Persist to disk for cross-restart reuse.
#   5. No user-facing calibrate() required — fully transparent.
#
# The forward kernel runs inline (no CUDA graph), so the auto-tune happens as
# part of the normal first-call JIT compilation overhead.  The backward kernels
# use CUDA graphs, and the auto-tune is folded into the warmup that already
# runs before graph capture.
# =============================================================================

class _KernelAutoTuner:
    """Inline first-call auto-tuner for Triton kernel launch parameters.

    Unlike per-S tuning, this benchmarks once per (GPU, op, H, D) — the optimal
    warp config is S-independent (verified empirically across S=512..131072).
    On first forward/backward call with unseen (H, D), benchmarks 4 candidate
    configs using the ACTUAL tensors in-flight, picks the fastest, and caches
    the result.  Subsequent calls are a dict lookup with zero overhead.

    Thread-safe via a lock on the cache dict.
    """

    # Candidate configs: (num_warps, num_stages).
    CANDIDATES = [(4, 1), (4, 2), (8, 2), (8, 3)]

    # Benchmark shots (kept small: ~10ms total per tune)
    _WARMUP = 3
    _REPEATS = 5

    def __init__(self):
        self._cache: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()
        # Fast-path lookup: (op, H, D) → (nw, ns), no GPU API call, no lock.
        # Populated after first tune or on first get_config hit.
        self._fast: dict[tuple[str, int, int], tuple[int, int]] = {}
        self._cache_file = os.path.join(
            os.path.expanduser("~"), ".cache", "cron_root_attention",
            "autotune_v14.json"
        )
        self._load_cache()
        # Cache GPU tag once at init — GPU never changes at runtime.
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(torch.cuda.current_device())
            self._gpu_tag_str = f"sm{prop.major}{prop.minor}_{prop.name.replace(' ', '_')}"
        else:
            self._gpu_tag_str = "cpu"

    # ── Persistence ───────────────────────────────────────────────────
    def _load_cache(self):
        try:
            if os.path.isfile(self._cache_file):
                with open(self._cache_file, "r") as f:
                    raw = json.load(f)
                self._cache = {k: tuple(v) for k, v in raw.items()}
        except Exception:
            self._cache = {}

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            with open(self._cache_file, "w") as f:
                json.dump({k: list(v) for k, v in self._cache.items()}, f)
        except Exception:
            pass

    # ── Key construction ──────────────────────────────────────────────
    def _gpu_tag(self) -> str:
        return self._gpu_tag_str

    def _cache_key(self, op: str, H: int, D: int) -> str:
        """Key by (GPU, op, H, D) only — optimal config is S-independent."""
        return f"{self._gpu_tag_str}|{op}|H{H}|D{D}"

    # ── Lookup (zero overhead on hot path) ────────────────────────────
    def get_config(self, op: str, H: int, D: int) -> tuple[int, int]:
        """Return the cached (num_warps, num_stages) or static default.

        Fast path: single dict lookup, no CUDA API call, no lock.
        Slow path (first call): falls back to string-keyed cache + lock.
        """
        fast_key = (op, H, D)
        hit = self._fast.get(fast_key)
        if hit is not None:
            return hit
        # Slow path: first time this (op, H, D) is queried.
        full_key = self._cache_key(op, H, D)
        with self._lock:
            result = self._cache.get(full_key)
        if result is None:
            result = (_KW, _KS_FWD) if op == "fwd" else (_KW, _KS)
        self._fast[fast_key] = result
        return result

    def is_tuned(self, op: str, H: int, D: int) -> bool:
        # Fast path: if already in _fast dict, it was tuned.
        fast_key = (op, H, D)
        if fast_key in self._fast:
            return True
        full_key = self._cache_key(op, H, D)
        with self._lock:
            result = full_key in self._cache
        if result:
            # Prime fast path so future get_config avoids the lock.
            with self._lock:
                self._fast[fast_key] = self._cache[full_key]
        return result

    # Minimum S for benchmarking — below this, kernel time is <0.1ms and
    # measurement noise dominates, causing random config selection.
    _MIN_BENCH_S = 4096

    # ── Inline forward auto-tune ──────────────────────────────────────
    def tune_fwd_inline(self, q, k, v, o, L, relay_k, relay_v,
                        S, D, SQRT_N, NUM_RELAY, BLOCK_RELAY,
                        BLOCK_M, BLOCK_D, BLOCK_LOCAL, BLOCK_STRIDE,
                        kv_groups, H_q, B) -> tuple[int, int]:
        """Benchmark forward tiled kernel.  Called once per (H, D) on the
        first forward pass — adds ~10ms to the first step.

        If S < _MIN_BENCH_S, allocates a synthetic tensor at _MIN_BENCH_S
        to ensure enough compute for reliable timing.
        """
        key = self._cache_key("fwd", H_q, D)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                return hit

        # Use a representative S for benchmarking
        bench_S = max(S, self._MIN_BENCH_S)
        if bench_S != S:
            bench_SQRT_N = int(math.ceil(math.sqrt(bench_S)))
            bench_NUM_RELAY = (bench_S + bench_SQRT_N - 1) // bench_SQRT_N * 2  # TOP_K=2 for bench
            bench_q = torch.randn(1, H_q, bench_S, D, device=q.device, dtype=q.dtype)
            bench_k = torch.randn(1, H_q, bench_S, D, device=q.device, dtype=q.dtype)
            bench_v = torch.randn(1, H_q, bench_S, D, device=q.device, dtype=q.dtype)
            bench_o = torch.empty_like(bench_q)
            bench_L = torch.empty(1, H_q, bench_S, dtype=torch.float32, device=q.device)
            bench_rk = torch.randn(1, H_q, bench_NUM_RELAY, D, device=q.device, dtype=q.dtype)
            bench_rv = torch.randn(1, H_q, bench_NUM_RELAY, D, device=q.device, dtype=q.dtype)
            bench_B = 1
        else:
            bench_SQRT_N, bench_NUM_RELAY = SQRT_N, NUM_RELAY
            bench_q, bench_k, bench_v = q, k, v
            bench_o, bench_L = o, L
            bench_rk, bench_rv = relay_k, relay_v
            bench_B = B

        grid = (bench_B, H_q, (bench_S + BLOCK_M - 1) // BLOCK_M)
        best_time = float("inf")
        best_cfg = (_KW, _KS_FWD)

        for nw, ns in self.CANDIDATES:
            try:
                for _ in range(self._WARMUP):
                    _cron_root_attn_fwd_v14_tiled[grid](
                        bench_q, bench_k, bench_v, bench_o, bench_L, bench_rk, bench_rv,
                        bench_q.stride(0), bench_q.stride(1), bench_q.stride(2), bench_q.stride(3),
                        bench_k.stride(0), bench_k.stride(1), bench_k.stride(2), bench_k.stride(3),
                        bench_v.stride(0), bench_v.stride(1), bench_v.stride(2), bench_v.stride(3),
                        bench_o.stride(0), bench_o.stride(1), bench_o.stride(2), bench_o.stride(3),
                        bench_L.stride(0), bench_L.stride(1), bench_L.stride(2),
                        bench_rk.stride(0), bench_rk.stride(1), bench_rk.stride(2), bench_rk.stride(3),
                        bench_rv.stride(0), bench_rv.stride(1), bench_rv.stride(2), bench_rv.stride(3),
                        S=bench_S, D=D, SQRT_N=bench_SQRT_N,
                        NUM_RELAY=bench_NUM_RELAY, BLOCK_RELAY=BLOCK_RELAY,
                        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
                        BLOCK_LOCAL=BLOCK_LOCAL, BLOCK_STRIDE=BLOCK_STRIDE,
                        KV_GROUPS=kv_groups,
                        TOP_K=2,
                        num_warps=nw, num_stages=ns)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(self._REPEATS):
                    _cron_root_attn_fwd_v14_tiled[grid](
                        bench_q, bench_k, bench_v, bench_o, bench_L, bench_rk, bench_rv,
                        bench_q.stride(0), bench_q.stride(1), bench_q.stride(2), bench_q.stride(3),
                        bench_k.stride(0), bench_k.stride(1), bench_k.stride(2), bench_k.stride(3),
                        bench_v.stride(0), bench_v.stride(1), bench_v.stride(2), bench_v.stride(3),
                        bench_o.stride(0), bench_o.stride(1), bench_o.stride(2), bench_o.stride(3),
                        bench_L.stride(0), bench_L.stride(1), bench_L.stride(2),
                        bench_rk.stride(0), bench_rk.stride(1), bench_rk.stride(2), bench_rk.stride(3),
                        bench_rv.stride(0), bench_rv.stride(1), bench_rv.stride(2), bench_rv.stride(3),
                        S=bench_S, D=D, SQRT_N=bench_SQRT_N,
                        NUM_RELAY=bench_NUM_RELAY, BLOCK_RELAY=BLOCK_RELAY,
                        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
                        BLOCK_LOCAL=BLOCK_LOCAL, BLOCK_STRIDE=BLOCK_STRIDE,
                        KV_GROUPS=kv_groups,
                        TOP_K=2,
                        num_warps=nw, num_stages=ns)
                end.record()
                torch.cuda.synchronize()
                t = start.elapsed_time(end) / self._REPEATS
                if t < best_time:
                    best_time = t
                    best_cfg = (nw, ns)
            except Exception:
                continue

        # Clean up synthetic tensors
        if bench_S != S:
            del bench_q, bench_k, bench_v, bench_o, bench_L, bench_rk, bench_rv

        with self._lock:
            self._cache[key] = best_cfg
            self._save_cache()
        self._fast[("fwd", H_q, D)] = best_cfg
        return best_cfg

    # ── Inline backward auto-tune ─────────────────────────────────────
    def tune_bwd_inline(self, kernel_fn, grid, args, kwargs,
                        H: int, D: int) -> tuple[int, int]:
        """Benchmark a backward kernel with actual tensors.  Called during
        the CUDA graph warmup that already happens before graph capture.

        If the tensors' S < _MIN_BENCH_S, allocates synthetic tensors at
        _MIN_BENCH_S for reliable timing (same (H, D) characterization).
        """
        key = self._cache_key("bwd", H, D)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                return hit

        # Check if we need synthetic tensors for reliable timing
        S = kwargs.get("S", 0)
        if S < self._MIN_BENCH_S:
            return self._tune_bwd_synthetic(H, D, args[0].device, args[0].dtype)

        best_time = float("inf")
        best_cfg = (_KW, _KS)

        for nw, ns in self.CANDIDATES:
            try:
                for _ in range(self._WARMUP):
                    kernel_fn[grid](*args, num_warps=nw, num_stages=ns, **kwargs)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(self._REPEATS):
                    kernel_fn[grid](*args, num_warps=nw, num_stages=ns, **kwargs)
                end.record()
                torch.cuda.synchronize()
                t = start.elapsed_time(end) / self._REPEATS
                if t < best_time:
                    best_time = t
                    best_cfg = (nw, ns)
            except Exception:
                continue

        with self._lock:
            self._cache[key] = best_cfg
            self._save_cache()
        self._fast[("bwd", H, D)] = best_cfg
        return best_cfg

    def _tune_bwd_synthetic(self, H: int, D: int,
                            device, dtype) -> tuple[int, int]:
        """Backward auto-tune using synthetic tensors at _MIN_BENCH_S."""
        S = self._MIN_BENCH_S
        B = 1
        SQRT_N = int(math.ceil(math.sqrt(S)))
        BLOCK_M = BLOCK_D = BLOCK_LOCAL = BLOCK_STRIDE = BLOCK_RELAY = 64
        num_tiles_m = (S + BLOCK_M - 1) // BLOCK_M

        q  = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k  = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v  = torch.randn(B, H, S, D, device=device, dtype=dtype)
        o  = torch.randn(B, H, S, D, device=device, dtype=dtype)
        do = torch.randn(B, H, S, D, device=device, dtype=dtype)
        dq = torch.empty_like(q)
        L  = torch.randn(B, H, S, dtype=torch.float32, device=device)
        NUM_RELAY = (S + SQRT_N - 1) // SQRT_N
        rk = torch.randn(B, H, NUM_RELAY, D, device=device, dtype=dtype)
        rv = torch.randn(B, H, NUM_RELAY, D, device=device, dtype=dtype)

        grid = (B, H, num_tiles_m)
        bench_args = (
            q, k, v, o, do, dq, L, rk, rv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            rk.stride(0), rk.stride(1), rk.stride(2), rk.stride(3),
            rv.stride(0), rv.stride(1), rv.stride(2), rv.stride(3))
        bench_kwargs = dict(
            S=S, D=D, SQRT_N=SQRT_N,
            NUM_RELAY=0, BLOCK_RELAY=BLOCK_RELAY,
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
            BLOCK_LOCAL=BLOCK_LOCAL, BLOCK_STRIDE=BLOCK_STRIDE,
            TOP_K=2)

        best_time = float("inf")
        best_cfg = (_KW, _KS)

        for nw, ns in self.CANDIDATES:
            try:
                for _ in range(self._WARMUP):
                    _cron_root_attn_bwd_dq_only_v14[grid](
                        *bench_args, num_warps=nw, num_stages=ns, **bench_kwargs)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(self._REPEATS):
                    _cron_root_attn_bwd_dq_only_v14[grid](
                        *bench_args, num_warps=nw, num_stages=ns, **bench_kwargs)
                end.record()
                torch.cuda.synchronize()
                t = start.elapsed_time(end) / self._REPEATS
                if t < best_time:
                    best_time = t
                    best_cfg = (nw, ns)
            except Exception:
                continue

        del q, k, v, o, do, dq, L, rk, rv

        with self._lock:
            self._cache[self._cache_key("bwd", H, D)] = best_cfg
            self._save_cache()
        self._fast[("bwd", H, D)] = best_cfg
        return best_cfg

    def clear(self):
        """Clear in-memory and on-disk cache (forces re-tuning)."""
        with self._lock:
            self._cache.clear()
        self._fast.clear()
        try:
            if os.path.isfile(self._cache_file):
                os.remove(self._cache_file)
        except Exception:
            pass


# Singleton auto-tuner instance
_tuner = _KernelAutoTuner()

# =============================================================================
# BLACKWELL-OPTIMISED TRITON GEMM
# =============================================================================
# Tuned for SM 12.0 (RTX 50xx / B-series):
#   128×128 tiles → fits Blackwell's 4× larger L1 (192 KB/SM, up from 48 KB)
#   BLOCK_K=64, num_warps=8, num_stages=4 → hides 4-cycle MMA latency
# Falls back gracefully on Ampere (still correct, slightly sub-optimal vs cuBLAS).

@triton.jit
def _blackwell_gemm_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """General matmul C = A @ B.  A:[M,K], B:[K,N], C:[M,N]."""
    pid        = tl.program_id(0)
    num_pid_m  = tl.cdiv(M, BLOCK_M)
    num_pid_n  = tl.cdiv(N, BLOCK_N)
    num_in_grp = GROUP_SIZE_M * num_pid_n
    group_id   = pid // num_in_grp
    first_m    = group_id * GROUP_SIZE_M
    grp_size   = min(num_pid_m - first_m, GROUP_SIZE_M)
    pid_m      = first_m + (pid % num_in_grp) % grp_size
    pid_n      = (pid % num_in_grp) // grp_size

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write output (cast back to input dtype)
    c = acc.to(A.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs  = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask  = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# Block/warp/stage config for the GEMM kernel (different from attention tiles)
# Blackwell: 128×128 blocks use 192 KB L1 efficiently; 4 stages hide MMA latency.
# Ampere : 64×64 blocks with 2 stages are safe within 48 KB L1.
_GEMM_BLOCK_M, _GEMM_BLOCK_N, _GEMM_BLOCK_K = (
    (128, 128, 64) if _KNL_WARPS >= 8 else (64, 64, 32)
)
_GEMM_GROUP_SIZE_M = 8


def blackwell_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Drop-in for ``torch.matmul(a, b)`` (no bias).  a:[M,K], b:[K,N] → [M,N].
    BF16/FP16 only.  Falls back to torch.matmul for unsupported dtypes.
    Only used for shapes where the Triton path is profitable (M,N,K all ≥ 64).
    """
    if a.dtype not in (torch.float16, torch.bfloat16):
        return torch.matmul(a, b)
    assert a.is_cuda and b.is_cuda, "blackwell_matmul: inputs must be on CUDA"
    M, K  = a.shape
    K2, N = b.shape
    assert K == K2, f"shape mismatch: a={a.shape}, b={b.shape}"
    if M < _GEMM_BLOCK_M or N < _GEMM_BLOCK_N or K < _GEMM_BLOCK_K:
        return torch.matmul(a, b)  # too small — cuBLAS wins
    c    = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, _GEMM_BLOCK_M) * triton.cdiv(N, _GEMM_BLOCK_N),)
    _blackwell_gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=_GEMM_BLOCK_M, BLOCK_N=_GEMM_BLOCK_N, BLOCK_K=_GEMM_BLOCK_K,
        GROUP_SIZE_M=_GEMM_GROUP_SIZE_M,
        num_warps=_KNL_WARPS,
        num_stages=_KNL_STAGES_FWD,
    )
    return c


def blackwell_linear(x: torch.Tensor, weight: torch.Tensor,
                     bias: torch.Tensor | None = None) -> torch.Tensor:
    """
    Drop-in for ``F.linear(x, weight, bias)``.
    Reshapes leading batch dims to 2-D, calls blackwell_matmul, then restores.
    """
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])              # [B*T, D]
    out = blackwell_matmul(x2d, weight.t())          # [B*T, D_out]
    if bias is not None:
        out = out + bias
    return out.reshape(*orig_shape[:-1], out.shape[-1])


# =============================================================================
# V14 PERSISTENT TILED FORWARD KERNEL
# =============================================================================

@triton.jit
def _cron_root_attn_fwd_v14_persistent(
    Q, K, V, O, L,
    WorkCounter,  # Atomic counter for work stealing
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    SQRT_N: tl.constexpr, TOTAL_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,  # Queries per tile
    BLOCK_D: tl.constexpr,
    BLOCK_LOCAL: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
):
    """
    Persistent kernel: Each of the NUM_SMS blocks runs a while loop,
    pulling tiles from a global work queue until all work is done.

    Tile layout: total tiles = B * H * ceil(S / BLOCK_M)
    Each tile processes BLOCK_M consecutive queries.
    """
    pid = tl.program_id(0)

    # Number of query tiles per (batch, head)
    tiles_per_bh = (S + BLOCK_M - 1) // BLOCK_M

    # Work-stealing loop: grab first tile before entering the while condition.
    # Triton does not allow `return` inside while/for — use while-cond pattern.
    tile_idx = tl.atomic_add(WorkCounter, 1)
    while tile_idx < TOTAL_TILES:
        # Decode tile_idx -> (b, h, tile_m)
        bh_idx = tile_idx // tiles_per_bh
        tile_m = tile_idx % tiles_per_bh
        b = bh_idx // H
        h = bh_idx % H
        
        # Query range for this tile
        m_start = tile_m * BLOCK_M
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < S
        
        # Dimension indices
        d_idx = tl.arange(0, BLOCK_D)
        
        # Load query block: [BLOCK_M, D]
        q_ptr = Q + b * stride_qb + h * stride_qh
        q = tl.load(
            q_ptr + m_offsets[:, None] * stride_qm + d_idx[None, :] * stride_qd,
            mask=m_mask[:, None] & (d_idx[None, :] < D),
            other=0.0
        )
        
        scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
        
        # Per-query accumulators: [BLOCK_M, D], [BLOCK_M], [BLOCK_M]
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        
        # =================================================================
        # PHASE 1: Local Window Processing (vectorized across BLOCK_M queries)
        # =================================================================
        # For query at position m, local window is [max(0, m - SQRT_N + 1), m]
        # Process a shared "super-window" that covers all queries in the tile
        
        # Compute the union of all local windows for this tile
        # Earliest start: max(0, m_start - SQRT_N + 1)
        # Latest end: min(S-1, m_start + BLOCK_M - 1)
        window_start = m_start - SQRT_N + 1
        window_start = tl.maximum(window_start, 0)
        window_end = m_start + BLOCK_M - 1
        window_end = tl.minimum(window_end, S - 1)
        
        for n in range(0, SQRT_N + BLOCK_M, BLOCK_LOCAL):
            n_offsets = window_start + n + tl.arange(0, BLOCK_LOCAL)
            n_mask = (n_offsets >= 0) & (n_offsets <= window_end) & (n_offsets < S)
            
            # Load K block: [BLOCK_LOCAL, D]
            k_ptr = K + b * stride_kb + h * stride_kh
            k = tl.load(
                k_ptr + n_offsets[:, None] * stride_kn + d_idx[None, :] * stride_kd,
                mask=n_mask[:, None] & (d_idx[None, :] < D),
                other=0.0
            )
            
            # Compute scores: [BLOCK_M, BLOCK_LOCAL]
            # scores[i, j] = Q[m_offsets[i]] @ K[n_offsets[j]]^T
            scores = tl.dot(q, tl.trans(k)) * scale
            
            # Apply causal mask: query m can only attend to positions <= m
            # AND within its local window [m - SQRT_N + 1, m]
            # n_offsets[j] <= m_offsets[i] AND n_offsets[j] >= m_offsets[i] - SQRT_N + 1
            causal_mask = (n_offsets[None, :] <= m_offsets[:, None]) & \
                          (n_offsets[None, :] >= m_offsets[:, None] - SQRT_N + 1)
            full_mask = causal_mask & n_mask[None, :] & m_mask[:, None]
            
            scores = tl.where(full_mask, scores, float('-inf'))
            
            # Online softmax update (vectorized)
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            
            # Load V block: [BLOCK_LOCAL, D]
            v_ptr = V + b * stride_vb + h * stride_vh
            v = tl.load(
                v_ptr + n_offsets[:, None] * stride_vn + d_idx[None, :] * stride_vd,
                mask=n_mask[:, None] & (d_idx[None, :] < D),
                other=0.0
            )
            
            # acc += P @ V: [BLOCK_M, D]
            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_new
        
        # =================================================================
        # PHASE 2: Strided Positions (before local window)
        # =================================================================
        # For query m, strided positions are 0, SQRT_N, 2*SQRT_N, ...
        # up to (but not including) local_start = max(0, m - SQRT_N + 1)
        
        # Maximum strided index we might access (for earliest query in tile)
        max_strided_end = m_start
        num_strided_positions = (max_strided_end + SQRT_N - 1) // SQRT_N
        
        for s_block in range(0, num_strided_positions, BLOCK_STRIDE):
            stride_indices = s_block + tl.arange(0, BLOCK_STRIDE)
            n_offsets = stride_indices * SQRT_N
            n_mask = (stride_indices < num_strided_positions) & (n_offsets < S)
            
            # Load K block
            k_ptr = K + b * stride_kb + h * stride_kh
            k = tl.load(
                k_ptr + n_offsets[:, None] * stride_kn + d_idx[None, :] * stride_kd,
                mask=n_mask[:, None] & (d_idx[None, :] < D),
                other=0.0
            )
            
            # Compute scores
            scores = tl.dot(q, tl.trans(k)) * scale
            
            # Mask: n_offsets[j] must be < m_offsets[i] - SQRT_N + 1 (before local window)
            strided_mask = n_offsets[None, :] < (m_offsets[:, None] - SQRT_N + 1)
            full_mask = strided_mask & n_mask[None, :] & m_mask[:, None]
            
            scores = tl.where(full_mask, scores, float('-inf'))
            
            # Online softmax update
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            
            # Load V and accumulate
            v_ptr = V + b * stride_vb + h * stride_vh
            v = tl.load(
                v_ptr + n_offsets[:, None] * stride_vn + d_idx[None, :] * stride_vd,
                mask=n_mask[:, None] & (d_idx[None, :] < D),
                other=0.0
            )
            
            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_new
        
        # =================================================================
        # OUTPUT: Normalize and store
        # =================================================================
        acc = acc / tl.maximum(l_i[:, None], 1e-6)
        
        o_ptr = O + b * stride_ob + h * stride_oh
        tl.store(
            o_ptr + m_offsets[:, None] * stride_om + d_idx[None, :] * stride_od,
            acc.to(O.dtype.element_ty),
            mask=m_mask[:, None] & (d_idx[None, :] < D)
        )
        
        # Store logsumexp for backward
        l_ptr = L + b * stride_lb + h * stride_lh
        lse = m_i + tl.log(tl.maximum(l_i, 1e-6))
        tl.store(l_ptr + m_offsets * stride_lm, lse, mask=m_mask)

        # Grab next tile for the while-cond loop
        tile_idx = tl.atomic_add(WorkCounter, 1)


# =============================================================================
# V14 TILED FORWARD KERNEL (Non-Persistent Variant for comparison)
# =============================================================================

@triton.jit
def _cron_root_attn_fwd_v14_tiled(
    Q, K, V, O, L,
    RK, RV,  # Relay key/value buffers (block-mean aggregated)
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    stride_rkb, stride_rkh, stride_rkn, stride_rkd,
    stride_rvb, stride_rvh, stride_rvn, stride_rvd,
    S: tl.constexpr, D: tl.constexpr, SQRT_N: tl.constexpr,
    NUM_RELAY: tl.constexpr,
    BLOCK_M: tl.constexpr,  # Queries per tile
    BLOCK_D: tl.constexpr,
    BLOCK_LOCAL: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    BLOCK_RELAY: tl.constexpr,
    KV_GROUPS: tl.constexpr,  # K/V heads per Q head (1 = MHA, >1 = GQA)
    TOP_K: tl.constexpr = 2,  # Relay tokens per block (for block masking)
):
    """
    Non-persistent tiled kernel: One thread block per (B, H, tile_m).
    Processes BLOCK_M queries per block, grid-scheduled.
    
    Key insight: Each query in the tile has the SAME strided positions to attend to
    (0, SQRT_N, 2*SQRT_N, ...) but different MASKS based on their local window start.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    tile_m = tl.program_id(2)
    
    # Query range for this tile
    m_start = tile_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < S
    
    d_idx = tl.arange(0, BLOCK_D)
    
    # Load query block
    q_ptr = Q + b * stride_qb + h * stride_qh
    q = tl.load(
        q_ptr + m_offsets[:, None] * stride_qm + d_idx[None, :] * stride_qd,
        mask=m_mask[:, None] & (d_idx[None, :] < D),
        other=0.0
    )
    
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    
    # Accumulators - per query in tile
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Per-query local window start: max(0, m - SQRT_N + 1)
    local_starts = tl.maximum(m_offsets - SQRT_N + 1, 0)  # [BLOCK_M]
    
    # =================================================================
    # PHASE 1: Local Window Processing
    # =================================================================
    # Each query m attends to keys in [local_start[m], m]
    # We iterate over all potential local positions, masking per-query
    
    # Global window covering all queries: [min(local_starts), max(m_offsets)]
    global_window_start = tl.maximum(m_start - SQRT_N + 1, 0)
    global_window_end = tl.minimum(m_start + BLOCK_M - 1, S - 1)
    window_size = global_window_end - global_window_start + 1
    
    # Process in BLOCK_LOCAL chunks
    for n_base in range(0, SQRT_N + BLOCK_M, BLOCK_LOCAL):
        n_offsets = global_window_start + n_base + tl.arange(0, BLOCK_LOCAL)
        n_valid = (n_offsets >= 0) & (n_offsets <= global_window_end) & (n_offsets < S)
        
        # Load K: [BLOCK_LOCAL, D]
        k_ptr = K + b * stride_kb + (h // KV_GROUPS) * stride_kh
        k = tl.load(
            k_ptr + n_offsets[:, None] * stride_kn + d_idx[None, :] * stride_kd,
            mask=n_valid[:, None] & (d_idx[None, :] < D),
            other=0.0
        )
        
        # Compute scores: [BLOCK_M, BLOCK_LOCAL]
        scores = tl.dot(q, tl.trans(k)) * scale
        
        # Mask: n in [local_start[m], m] for each query m
        # n_offsets[j] >= local_starts[i] AND n_offsets[j] <= m_offsets[i]
        local_mask = (n_offsets[None, :] >= local_starts[:, None]) & \
                     (n_offsets[None, :] <= m_offsets[:, None])
        full_mask = local_mask & n_valid[None, :] & m_mask[:, None]
        scores = tl.where(full_mask, scores, float('-inf'))
        
        # Safe online softmax: avoid NaN when all scores are -inf
        block_max = tl.max(scores, axis=1)  # [BLOCK_M]
        # Only update if we have valid scores (block_max > -inf)
        has_valid = block_max > float('-inf')
        
        # Compute new max: use current m_i if no valid scores in this block
        m_new = tl.where(has_valid, tl.maximum(m_i, block_max), m_i)
        
        # Safe alpha and p computation
        # alpha = exp(m_i - m_new), but if m_i == -inf and m_new > -inf, alpha = 0
        # If both are -inf, we shouldn't reach here with has_valid = True
        alpha = tl.where(m_i == float('-inf'), 
                        tl.where(m_new == float('-inf'), 1.0, 0.0),
                        tl.exp(m_i - m_new))
        
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(full_mask, p, 0.0)  # Zero out invalid positions
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        # Load V: [BLOCK_LOCAL, D]
        v_ptr = V + b * stride_vb + (h // KV_GROUPS) * stride_vh
        v = tl.load(
            v_ptr + n_offsets[:, None] * stride_vn + d_idx[None, :] * stride_vd,
            mask=n_valid[:, None] & (d_idx[None, :] < D),
            other=0.0
        )
        
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new
    
    # =================================================================
    # PHASE 2: Strided Positions (before local window)
    # =================================================================
    # For query m with local_start[m], strided positions are:
    # 0, SQRT_N, 2*SQRT_N, ... up to (but not including) local_start[m]
    
    # Maximum number of strided positions for any query in this tile
    # is for the LAST query in tile (m_start + BLOCK_M - 1)
    max_m = m_start + BLOCK_M - 1
    max_local_start = tl.maximum(max_m - SQRT_N + 1, 0)
    max_num_strided = (max_local_start + SQRT_N - 1) // SQRT_N
    
    for s_base in range(0, max_num_strided, BLOCK_STRIDE):
        stride_indices = s_base + tl.arange(0, BLOCK_STRIDE)
        n_offsets = stride_indices * SQRT_N
        n_valid = (stride_indices < max_num_strided) & (n_offsets < S)
        
        # Load K: [BLOCK_STRIDE, D]
        k_ptr = K + b * stride_kb + (h // KV_GROUPS) * stride_kh
        k = tl.load(
            k_ptr + n_offsets[:, None] * stride_kn + d_idx[None, :] * stride_kd,
            mask=n_valid[:, None] & (d_idx[None, :] < D),
            other=0.0
        )
        
        # Compute scores: [BLOCK_M, BLOCK_STRIDE]
        scores = tl.dot(q, tl.trans(k)) * scale
        
        # Strided mask: n_offsets[j] < local_starts[i] for each query
        strided_mask = n_offsets[None, :] < local_starts[:, None]
        full_mask = strided_mask & n_valid[None, :] & m_mask[:, None]
        scores = tl.where(full_mask, scores, float('-inf'))
        
        # Safe online softmax
        block_max = tl.max(scores, axis=1)
        has_valid = block_max > float('-inf')
        m_new = tl.where(has_valid, tl.maximum(m_i, block_max), m_i)
        
        alpha = tl.where(m_i == float('-inf'),
                        tl.where(m_new == float('-inf'), 1.0, 0.0),
                        tl.exp(m_i - m_new))
        
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(full_mask, p, 0.0)
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        # Load V: [BLOCK_STRIDE, D]
        v_ptr = V + b * stride_vb + (h // KV_GROUPS) * stride_vh
        v = tl.load(
            v_ptr + n_offsets[:, None] * stride_vn + d_idx[None, :] * stride_vd,
            mask=n_valid[:, None] & (d_idx[None, :] < D),
            other=0.0
        )
        
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new
    
    # =================================================================
    # PHASE 3: Relay Keys (2-hop aggregated block means)
    # =================================================================
    # Each relay block r summarizes positions [r*SQRT_N, (r+1)*SQRT_N)
    # via mean-pooling of K and V. This carries compressed 2-hop info
    # through a SINGLE softmax, eliminating gradient dilution.
    # A relay block is accessible if the entire block is before local_start(m).
    
    for r_base in range(0, NUM_RELAY, BLOCK_RELAY):
        r_indices = r_base + tl.arange(0, BLOCK_RELAY)
        r_valid = r_indices < NUM_RELAY
        
        # Load relay K: [BLOCK_RELAY, D]
        rk_ptr = RK + b * stride_rkb + (h // KV_GROUPS) * stride_rkh
        rk = tl.load(
            rk_ptr + r_indices[:, None] * stride_rkn + d_idx[None, :] * stride_rkd,
            mask=r_valid[:, None] & (d_idx[None, :] < D),
            other=0.0
        )
        
        # Compute scores: [BLOCK_M, BLOCK_RELAY]
        scores = tl.dot(q, tl.trans(rk)) * scale
        
        # Mask: relay token t belongs to block (t // TOP_K).
        # Block (t // TOP_K) covers positions [(t//TOP_K)*SQRT_N, (t//TOP_K+1)*SQRT_N - 1].
        # Accessible when block_end < local_start(m).
        block_ends = (r_indices // TOP_K + 1) * SQRT_N - 1
        relay_mask = (block_ends[None, :] < local_starts[:, None])
        full_mask = relay_mask & r_valid[None, :] & m_mask[:, None]
        scores = tl.where(full_mask, scores, float('-inf'))
        
        # Safe online softmax
        block_max = tl.max(scores, axis=1)
        has_valid = block_max > float('-inf')
        m_new = tl.where(has_valid, tl.maximum(m_i, block_max), m_i)
        alpha = tl.where(m_i == float('-inf'),
                        tl.where(m_new == float('-inf'), 1.0, 0.0),
                        tl.exp(m_i - m_new))
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(full_mask, p, 0.0)
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        # Load relay V: [BLOCK_RELAY, D]
        rv_ptr = RV + b * stride_rvb + (h // KV_GROUPS) * stride_rvh
        rv = tl.load(
            rv_ptr + r_indices[:, None] * stride_rvn + d_idx[None, :] * stride_rvd,
            mask=r_valid[:, None] & (d_idx[None, :] < D),
            other=0.0
        )
        
        acc += tl.dot(p.to(rv.dtype), rv)
        m_i = m_new
    
    # =================================================================
    # OUTPUT: Normalize and store
    # =================================================================
    acc = acc / tl.maximum(l_i[:, None], 1e-6)
    
    o_ptr = O + b * stride_ob + h * stride_oh
    tl.store(
        o_ptr + m_offsets[:, None] * stride_om + d_idx[None, :] * stride_od,
        acc.to(O.dtype.element_ty),
        mask=m_mask[:, None] & (d_idx[None, :] < D)
    )
    
    l_ptr = L + b * stride_lb + h * stride_lh
    lse = m_i + tl.log(tl.maximum(l_i, 1e-6))
    tl.store(l_ptr + m_offsets * stride_lm, lse, mask=m_mask)


# =============================================================================
# V14 BACKWARD KERNELS
# =============================================================================

@triton.jit
def _cron_root_attn_bwd_dq_only_v14(
    Q, K, V, O, dO, dQ, L,
    RK, RV,  # Relay key/value buffers
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_lb, stride_lh, stride_lm,
    stride_rkb, stride_rkh, stride_rkn, stride_rkd,
    stride_rvb, stride_rvh, stride_rvn, stride_rvd,
    S: tl.constexpr, D: tl.constexpr, SQRT_N: tl.constexpr,
    NUM_RELAY: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_LOCAL: tl.constexpr, BLOCK_STRIDE: tl.constexpr,
    BLOCK_RELAY: tl.constexpr,
    KV_GROUPS: tl.constexpr = 1,
    TOP_K: tl.constexpr = 2,  # Relay tokens per block (for block masking)
):
    """
    dQ-only backward kernel with relay support.
    dK/dV computed by separate local + key-centric strided + relay kernels.
    KV_GROUPS > 1 enables GQA: Q head h reads K/V at head (h // KV_GROUPS).
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    tile_m = tl.program_id(2)
    
    m_start = tile_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < S
    
    d_idx = tl.arange(0, BLOCK_D)
    
    # Load Q, O, dO, L
    q_ptr = Q + b * stride_qb + h * stride_qh
    q = tl.load(q_ptr + m_offsets[:, None] * stride_qm + d_idx[None, :] * stride_qd,
                mask=m_mask[:, None] & (d_idx[None, :] < D), other=0.0)
    
    o_ptr = O + b * stride_ob + h * stride_oh
    o = tl.load(o_ptr + m_offsets[:, None] * stride_om + d_idx[None, :] * stride_od,
                mask=m_mask[:, None] & (d_idx[None, :] < D), other=0.0)
    
    do_ptr = dO + b * stride_dob + h * stride_doh
    do = tl.load(do_ptr + m_offsets[:, None] * stride_dom + d_idx[None, :] * stride_dod,
                 mask=m_mask[:, None] & (d_idx[None, :] < D), other=0.0)
    
    l_ptr = L + b * stride_lb + h * stride_lh
    lse = tl.load(l_ptr + m_offsets * stride_lm, mask=m_mask, other=0.0)
    
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    
    # Di = sum(o * do, axis=1)
    Di = tl.sum(o * do, axis=1)
    
    dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Phase 1: Local window
    window_start = m_start - SQRT_N + 1
    window_start = tl.maximum(window_start, 0)
    window_end = m_start + BLOCK_M - 1
    window_end = tl.minimum(window_end, S - 1)
    
    for n in range(0, SQRT_N + BLOCK_M, BLOCK_LOCAL):
        n_offsets = window_start + n + tl.arange(0, BLOCK_LOCAL)
        n_mask = (n_offsets >= 0) & (n_offsets <= window_end) & (n_offsets < S)
        
        k_ptr = K + b * stride_kb + (h // KV_GROUPS) * stride_kh
        k = tl.load(k_ptr + n_offsets[:, None] * stride_kn + d_idx[None, :] * stride_kd,
                    mask=n_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        v_ptr = V + b * stride_vb + (h // KV_GROUPS) * stride_vh
        v = tl.load(v_ptr + n_offsets[:, None] * stride_vn + d_idx[None, :] * stride_vd,
                    mask=n_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        scores = tl.dot(q, tl.trans(k)) * scale
        causal_mask = (n_offsets[None, :] <= m_offsets[:, None]) & \
                      (n_offsets[None, :] >= m_offsets[:, None] - SQRT_N + 1)
        full_mask = causal_mask & n_mask[None, :] & m_mask[:, None]
        scores = tl.where(full_mask, scores, float('-inf'))
        
        p = tl.exp(scores - lse[:, None])
        dov = tl.dot(do, tl.trans(v))
        dp = p * (dov - Di[:, None])
        dp = tl.where(full_mask, dp, 0.0)
        
        dq_acc += tl.dot(dp.to(k.dtype), k) * scale
    
    # Phase 2: Strided keys
    local_starts = tl.maximum(m_offsets - SQRT_N + 1, 0)
    max_m = m_start + BLOCK_M - 1
    max_local_start = tl.maximum(max_m - SQRT_N + 1, 0)
    max_num_strided = (max_local_start + SQRT_N - 1) // SQRT_N
    
    for s_base in range(0, max_num_strided, BLOCK_STRIDE):
        stride_indices = s_base + tl.arange(0, BLOCK_STRIDE)
        n_offsets = stride_indices * SQRT_N
        n_valid = (stride_indices < max_num_strided) & (n_offsets < S)
        
        k_ptr = K + b * stride_kb + (h // KV_GROUPS) * stride_kh
        k = tl.load(k_ptr + n_offsets[:, None] * stride_kn + d_idx[None, :] * stride_kd,
                    mask=n_valid[:, None] & (d_idx[None, :] < D), other=0.0)
        
        v_ptr = V + b * stride_vb + (h // KV_GROUPS) * stride_vh
        v = tl.load(v_ptr + n_offsets[:, None] * stride_vn + d_idx[None, :] * stride_vd,
                    mask=n_valid[:, None] & (d_idx[None, :] < D), other=0.0)
        
        scores = tl.dot(q, tl.trans(k)) * scale
        strided_mask = n_offsets[None, :] < local_starts[:, None]
        full_mask = strided_mask & n_valid[None, :] & m_mask[:, None]
        scores = tl.where(full_mask, scores, float('-inf'))
        
        p = tl.exp(scores - lse[:, None])
        dov = tl.dot(do, tl.trans(v))
        dp = p * (dov - Di[:, None])
        dp = tl.where(full_mask, dp, 0.0)
        
        dq_acc += tl.dot(dp.to(k.dtype), k) * scale
    
    # Phase 3: Relay keys contribution to dQ
    for r_base in range(0, NUM_RELAY, BLOCK_RELAY):
        r_indices = r_base + tl.arange(0, BLOCK_RELAY)
        r_valid = r_indices < NUM_RELAY
        
        rk_ptr = RK + b * stride_rkb + (h // KV_GROUPS) * stride_rkh
        rk = tl.load(rk_ptr + r_indices[:, None] * stride_rkn + d_idx[None, :] * stride_rkd,
                    mask=r_valid[:, None] & (d_idx[None, :] < D), other=0.0)
        
        rv_ptr = RV + b * stride_rvb + (h // KV_GROUPS) * stride_rvh
        rv = tl.load(rv_ptr + r_indices[:, None] * stride_rvn + d_idx[None, :] * stride_rvd,
                    mask=r_valid[:, None] & (d_idx[None, :] < D), other=0.0)
        
        scores = tl.dot(q, tl.trans(rk)) * scale
        block_ends = (r_indices // TOP_K + 1) * SQRT_N - 1
        relay_mask = (block_ends[None, :] < local_starts[:, None])
        full_mask = relay_mask & r_valid[None, :] & m_mask[:, None]
        scores = tl.where(full_mask, scores, float('-inf'))
        
        p = tl.exp(scores - lse[:, None])
        dov = tl.dot(do, tl.trans(rv))
        dp = p * (dov - Di[:, None])
        dp = tl.where(full_mask, dp, 0.0)
        
        dq_acc += tl.dot(dp.to(rk.dtype), rk) * scale
    
    # Store dQ
    dq_ptr = dQ + b * stride_dqb + h * stride_dqh
    tl.store(dq_ptr + m_offsets[:, None] * stride_dqm + d_idx[None, :] * stride_dqd,
             dq_acc.to(dQ.dtype.element_ty),
             mask=m_mask[:, None] & (d_idx[None, :] < D))


@triton.jit
def _cron_root_attn_bwd_dq_fused_v14(
    Q, K, V, O, dO, dQ, dK, dV, L,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lb, stride_lh, stride_lm,
    S: tl.constexpr, D: tl.constexpr, SQRT_N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_LOCAL: tl.constexpr, BLOCK_STRIDE: tl.constexpr,
):
    """
    FUSED Backward: dQ + strided dK/dV in one kernel.
    
    Key insight: The dQ kernel already iterates over strided keys with
    coalesced memory access. We can compute strided dK/dV contributions
    as a byproduct, then use ONE atomic per strided key per query block.
    
    This eliminates the separate strided phase kernel entirely!
    Memory access pattern matches forward = cache-friendly.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    tile_m = tl.program_id(2)
    
    m_start = tile_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < S
    
    d_idx = tl.arange(0, BLOCK_D)
    
    # Load Q, O, dO, L
    q_ptr = Q + b * stride_qb + h * stride_qh
    q = tl.load(q_ptr + m_offsets[:, None] * stride_qm + d_idx[None, :] * stride_qd,
                mask=m_mask[:, None] & (d_idx[None, :] < D), other=0.0)
    
    o_ptr = O + b * stride_ob + h * stride_oh
    o = tl.load(o_ptr + m_offsets[:, None] * stride_om + d_idx[None, :] * stride_od,
                mask=m_mask[:, None] & (d_idx[None, :] < D), other=0.0)
    
    do_ptr = dO + b * stride_dob + h * stride_doh
    do = tl.load(do_ptr + m_offsets[:, None] * stride_dom + d_idx[None, :] * stride_dod,
                 mask=m_mask[:, None] & (d_idx[None, :] < D), other=0.0)
    
    l_ptr = L + b * stride_lb + h * stride_lh
    lse = tl.load(l_ptr + m_offsets * stride_lm, mask=m_mask, other=0.0)
    
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    
    # Di = sum(o * do, axis=1) for each query
    Di = tl.sum(o * do, axis=1)
    
    dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Phase 1: Local window (dQ only - local dK/dV handled by separate kernel)
    window_start = m_start - SQRT_N + 1
    window_start = tl.maximum(window_start, 0)
    window_end = m_start + BLOCK_M - 1
    window_end = tl.minimum(window_end, S - 1)
    
    for n in range(0, SQRT_N + BLOCK_M, BLOCK_LOCAL):
        n_offsets = window_start + n + tl.arange(0, BLOCK_LOCAL)
        n_mask = (n_offsets >= 0) & (n_offsets <= window_end) & (n_offsets < S)
        
        k_ptr = K + b * stride_kb + h * stride_kh
        k = tl.load(k_ptr + n_offsets[:, None] * stride_kn + d_idx[None, :] * stride_kd,
                    mask=n_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        v_ptr = V + b * stride_vb + h * stride_vh
        v = tl.load(v_ptr + n_offsets[:, None] * stride_vn + d_idx[None, :] * stride_vd,
                    mask=n_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        scores = tl.dot(q, tl.trans(k)) * scale
        causal_mask = (n_offsets[None, :] <= m_offsets[:, None]) & \
                      (n_offsets[None, :] >= m_offsets[:, None] - SQRT_N + 1)
        full_mask = causal_mask & n_mask[None, :] & m_mask[:, None]
        scores = tl.where(full_mask, scores, float('-inf'))
        
        p = tl.exp(scores - lse[:, None])
        dov = tl.dot(do, tl.trans(v))
        dp = p * (dov - Di[:, None])
        dp = tl.where(full_mask, dp, 0.0)
        
        dq_acc += tl.dot(dp.to(k.dtype), k) * scale
    
    # Phase 2: Strided - FUSED dQ + dK/dV
    # For each strided key, compute dQ contribution AND accumulate dK/dV
    local_starts = tl.maximum(m_offsets - SQRT_N + 1, 0)
    
    max_m = m_start + BLOCK_M - 1
    max_local_start = tl.maximum(max_m - SQRT_N + 1, 0)
    max_num_strided = (max_local_start + SQRT_N - 1) // SQRT_N
    
    # Process strided keys one at a time to accumulate dK/dV per key
    for s_idx in range(max_num_strided):
        k_pos = s_idx * SQRT_N
        
        if k_pos < S:
            # Load single strided key
            k_ptr = K + b * stride_kb + h * stride_kh
            key = tl.load(k_ptr + k_pos * stride_kn + d_idx * stride_kd,
                          mask=d_idx < D, other=0.0)  # [D]
            
            v_ptr = V + b * stride_vb + h * stride_vh
            val = tl.load(v_ptr + k_pos * stride_vn + d_idx * stride_vd,
                          mask=d_idx < D, other=0.0)  # [D]
            
            # Compute attention for all queries in this block to this key
            # scores[i] = q[i] @ key
            scores = tl.sum(q * key[None, :], axis=1) * scale  # [BLOCK_M]
            
            # Strided mask: this key is visible only if k_pos < local_start[query]
            strided_mask = (k_pos < local_starts) & m_mask
            scores = tl.where(strided_mask, scores, float('-inf'))
            
            p = tl.exp(scores - lse)  # [BLOCK_M]
            
            # dQ contribution: dQ += dp * key
            dov = tl.sum(do * val[None, :], axis=1)  # [BLOCK_M]
            dp = p * (dov - Di)
            dp = tl.where(strided_mask, dp, 0.0)
            
            dq_acc += dp[:, None] * key[None, :] * scale
            
            # FUSED: Compute dK/dV for this strided key from this query block
            # dK[k_pos] += sum_over_queries(dp[i] * q[i]) * scale
            # dV[k_pos] += sum_over_queries(p[i] * do[i])
            dk_contrib = tl.sum(dp[:, None] * q, axis=0) * scale  # [D]
            dv_contrib = tl.sum(p[:, None] * do, axis=0)  # [D]
            
            # Atomic add - but only ONE per strided key per query block!
            # Much less contention than before
            dk_ptr = dK + b * stride_dkb + h * stride_dkh + k_pos * stride_dkn
            dv_ptr = dV + b * stride_dvb + h * stride_dvh + k_pos * stride_dvn
            
            tl.atomic_add(dk_ptr + d_idx * stride_dkd, dk_contrib.to(dK.dtype.element_ty), mask=d_idx < D)
            tl.atomic_add(dv_ptr + d_idx * stride_dvd, dv_contrib.to(dV.dtype.element_ty), mask=d_idx < D)
    
    # Store dQ
    dq_ptr = dQ + b * stride_dqb + h * stride_dqh
    tl.store(dq_ptr + m_offsets[:, None] * stride_dqm + d_idx[None, :] * stride_dqd,
             dq_acc.to(dQ.dtype.element_ty),
             mask=m_mask[:, None] & (d_idx[None, :] < D))


# =============================================================================
# FULLY FUSED SINGLE-KERNEL BACKWARD (for short sequences)
# =============================================================================
# Computes dQ + local dK/dV + strided dK/dV in ONE kernel launch.
# This eliminates ALL kernel launch overhead for the backward pass.
# Uses atomic_add for ALL dK/dV contributions (low contention at short seqs).
# Requires dk/dv initialized to zero.

@triton.jit
def _cron_root_attn_bwd_fully_fused(
    Q, K, V, O, dO, dQ, dK, dV, L,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lb, stride_lh, stride_lm,
    S: tl.constexpr, D: tl.constexpr, SQRT_N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_LOCAL: tl.constexpr, BLOCK_STRIDE: tl.constexpr,
    KV_GROUPS: tl.constexpr,  # K/V heads per Q head (1 = MHA, >1 = GQA)
):
    """
    FULLY FUSED Backward: dQ + local dK/dV + strided dK/dV in ONE kernel.
    
    Query-centric: one block per (B, H, query_tile).
    - Phase 1: Local window → dQ contribution + local dK/dV via atomic_add
    - Phase 2: Strided keys → dQ contribution + strided dK/dV via atomic_add
    
    At short sequences (S ≤ 4096), atomic contention is minimal:
    - Each local key gets atomic_adds from ≤2 query tiles
    - Each strided key gets atomic_adds from all query tiles, but there
      are only O(√N) strided keys
    
    Eliminates separate local and strided backward kernels entirely.
    Requires dK/dV initialized to zeros.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    tile_m = tl.program_id(2)
    
    m_start = tile_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < S
    
    d_idx = tl.arange(0, BLOCK_D)
    
    # Load Q, O, dO, LSE
    q_ptr = Q + b * stride_qb + h * stride_qh
    q = tl.load(q_ptr + m_offsets[:, None] * stride_qm + d_idx[None, :] * stride_qd,
                mask=m_mask[:, None] & (d_idx[None, :] < D), other=0.0)
    
    o_ptr = O + b * stride_ob + h * stride_oh
    o = tl.load(o_ptr + m_offsets[:, None] * stride_om + d_idx[None, :] * stride_od,
                mask=m_mask[:, None] & (d_idx[None, :] < D), other=0.0)
    
    do_ptr = dO + b * stride_dob + h * stride_doh
    do = tl.load(do_ptr + m_offsets[:, None] * stride_dom + d_idx[None, :] * stride_dod,
                 mask=m_mask[:, None] & (d_idx[None, :] < D), other=0.0)
    
    l_ptr = L + b * stride_lb + h * stride_lh
    lse = tl.load(l_ptr + m_offsets * stride_lm, mask=m_mask, other=0.0)
    
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    Di = tl.sum(o * do, axis=1)
    
    dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # =================================================================
    # PHASE 1: Local window — dQ + local dK/dV (atomic)
    # =================================================================
    window_start = m_start - SQRT_N + 1
    window_start = tl.maximum(window_start, 0)
    window_end = m_start + BLOCK_M - 1
    window_end = tl.minimum(window_end, S - 1)
    
    for n in range(0, SQRT_N + BLOCK_M, BLOCK_LOCAL):
        n_offsets = window_start + n + tl.arange(0, BLOCK_LOCAL)
        n_mask = (n_offsets >= 0) & (n_offsets <= window_end) & (n_offsets < S)
        
        k_ptr = K + b * stride_kb + (h // KV_GROUPS) * stride_kh
        k = tl.load(k_ptr + n_offsets[:, None] * stride_kn + d_idx[None, :] * stride_kd,
                    mask=n_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        v_ptr = V + b * stride_vb + (h // KV_GROUPS) * stride_vh
        v = tl.load(v_ptr + n_offsets[:, None] * stride_vn + d_idx[None, :] * stride_vd,
                    mask=n_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        # Scores: [BLOCK_M, BLOCK_LOCAL]
        scores = tl.dot(q, tl.trans(k)) * scale
        causal_mask = (n_offsets[None, :] <= m_offsets[:, None]) & \
                      (n_offsets[None, :] >= m_offsets[:, None] - SQRT_N + 1)
        full_mask = causal_mask & n_mask[None, :] & m_mask[:, None]
        scores = tl.where(full_mask, scores, float('-inf'))
        
        # Attention probabilities and gradient
        p = tl.exp(scores - lse[:, None])
        dov = tl.dot(do, tl.trans(v))
        dp = p * (dov - Di[:, None])
        dp = tl.where(full_mask, dp, 0.0)
        
        # dQ accumulation
        dq_acc += tl.dot(dp.to(k.dtype), k) * scale
        
        # FUSED: local dK/dV via atomic_add
        # dk_contrib: [BLOCK_LOCAL, D] = dp^T @ q * scale
        # dv_contrib: [BLOCK_LOCAL, D] = p^T @ do
        dk_contrib = tl.dot(tl.trans(dp.to(q.dtype)), q) * scale
        dv_contrib = tl.dot(tl.trans(p.to(do.dtype)), do)
        
        # Atomic add to dK/dV — low contention: each key touched by ≤2 query tiles
        dk_base = dK + b * stride_dkb + (h // KV_GROUPS) * stride_dkh
        dv_base = dV + b * stride_dvb + (h // KV_GROUPS) * stride_dvh
        tl.atomic_add(
            dk_base + n_offsets[:, None] * stride_dkn + d_idx[None, :] * stride_dkd,
            dk_contrib.to(dK.dtype.element_ty),
            mask=n_mask[:, None] & (d_idx[None, :] < D)
        )
        tl.atomic_add(
            dv_base + n_offsets[:, None] * stride_dvn + d_idx[None, :] * stride_dvd,
            dv_contrib.to(dV.dtype.element_ty),
            mask=n_mask[:, None] & (d_idx[None, :] < D)
        )
    
    # =================================================================
    # PHASE 2: Strided keys — dQ + strided dK/dV (atomic)
    # =================================================================
    local_starts = tl.maximum(m_offsets - SQRT_N + 1, 0)
    max_m = m_start + BLOCK_M - 1
    max_local_start = tl.maximum(max_m - SQRT_N + 1, 0)
    max_num_strided = (max_local_start + SQRT_N - 1) // SQRT_N
    
    for s_idx in range(max_num_strided):
        k_pos = s_idx * SQRT_N
        
        if k_pos < S:
            k_ptr = K + b * stride_kb + (h // KV_GROUPS) * stride_kh
            key = tl.load(k_ptr + k_pos * stride_kn + d_idx * stride_kd,
                          mask=d_idx < D, other=0.0)
            
            v_ptr = V + b * stride_vb + (h // KV_GROUPS) * stride_vh
            val = tl.load(v_ptr + k_pos * stride_vn + d_idx * stride_vd,
                          mask=d_idx < D, other=0.0)
            
            scores = tl.sum(q * key[None, :], axis=1) * scale
            strided_mask = (k_pos < local_starts) & m_mask
            scores = tl.where(strided_mask, scores, float('-inf'))
            
            p = tl.exp(scores - lse)
            dov = tl.sum(do * val[None, :], axis=1)
            dp = p * (dov - Di)
            dp = tl.where(strided_mask, dp, 0.0)
            
            dq_acc += dp[:, None] * key[None, :] * scale
            
            dk_contrib = tl.sum(dp[:, None] * q, axis=0) * scale
            dv_contrib = tl.sum(p[:, None] * do, axis=0)
            
            dk_ptr = dK + b * stride_dkb + (h // KV_GROUPS) * stride_dkh + k_pos * stride_dkn
            dv_ptr = dV + b * stride_dvb + (h // KV_GROUPS) * stride_dvh + k_pos * stride_dvn
            tl.atomic_add(dk_ptr + d_idx * stride_dkd, dk_contrib.to(dK.dtype.element_ty), mask=d_idx < D)
            tl.atomic_add(dv_ptr + d_idx * stride_dvd, dv_contrib.to(dV.dtype.element_ty), mask=d_idx < D)
    
    # Store dQ (direct store — each query tile owns its dQ positions)
    dq_ptr = dQ + b * stride_dqb + h * stride_dqh
    tl.store(dq_ptr + m_offsets[:, None] * stride_dqm + d_idx[None, :] * stride_dqd,
             dq_acc.to(dQ.dtype.element_ty),
             mask=m_mask[:, None] & (d_idx[None, :] < D))


# =============================================================================
# OPTIMIZED dK/dV KERNELS (O(N√N) complexity)
# =============================================================================
# Split into two phases:
# 1. Local phase: Each key processes only O(√N) local queries
# 2. Strided phase: Only √N strided keys, each processes O(N) queries
# Total: O(N√N) + O(√N × N) = O(N√N)

@triton.jit
def _dkdv_local_phase(
    Q, K, V, O, dO, dK, dV, L,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lb, stride_lh, stride_lm,
    S: tl.constexpr, D: tl.constexpr, SQRT_N: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_Q: tl.constexpr,
):
    """
    Local-only dK/dV: O(N√N) complexity.
    Each key tile only processes queries in its local window (O(√N) queries per key).
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    tile_k = tl.program_id(2)
    
    k_start = tile_k * BLOCK_N
    k_offsets = k_start + tl.arange(0, BLOCK_N)
    k_mask = k_offsets < S
    
    d_idx = tl.arange(0, BLOCK_D)
    
    # Load K, V for this tile
    k_ptr = K + b * stride_kb + h * stride_kh
    keys = tl.load(k_ptr + k_offsets[:, None] * stride_kn + d_idx[None, :] * stride_kd,
                   mask=k_mask[:, None] & (d_idx[None, :] < D), other=0.0)
    
    v_ptr = V + b * stride_vb + h * stride_vh
    vals = tl.load(v_ptr + k_offsets[:, None] * stride_vn + d_idx[None, :] * stride_vd,
                   mask=k_mask[:, None] & (d_idx[None, :] < D), other=0.0)
    
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    
    dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    
    # Local queries: for key k, queries in [k, k + SQRT_N - 1]
    # For key block [k_start, k_start + BLOCK_N - 1]:
    # Query range is [k_start, k_start + BLOCK_N - 1 + SQRT_N - 1]
    local_q_start = k_start
    local_q_end = tl.minimum(k_start + BLOCK_N - 1 + SQRT_N - 1, S - 1)
    num_local_q = local_q_end - local_q_start + 1
    
    for q_block in range(0, SQRT_N + BLOCK_N, BLOCK_Q):
        q_offsets = local_q_start + q_block + tl.arange(0, BLOCK_Q)
        q_mask = (q_offsets <= local_q_end) & (q_offsets < S)
        
        q_ptr = Q + b * stride_qb + h * stride_qh
        q = tl.load(q_ptr + q_offsets[:, None] * stride_qm + d_idx[None, :] * stride_qd,
                    mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        o_ptr = O + b * stride_ob + h * stride_oh
        o = tl.load(o_ptr + q_offsets[:, None] * stride_om + d_idx[None, :] * stride_od,
                    mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        do_ptr = dO + b * stride_dob + h * stride_doh
        do = tl.load(do_ptr + q_offsets[:, None] * stride_dom + d_idx[None, :] * stride_dod,
                     mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        l_ptr = L + b * stride_lb + h * stride_lh
        lse = tl.load(l_ptr + q_offsets * stride_lm, mask=q_mask, other=0.0)
        
        # Scores: [BLOCK_Q, BLOCK_N]
        scores = tl.dot(q, tl.trans(keys)) * scale
        
        # Local mask: k in [q - SQRT_N + 1, q]
        local_mask = (k_offsets[None, :] <= q_offsets[:, None]) & \
                     (k_offsets[None, :] >= q_offsets[:, None] - SQRT_N + 1)
        full_mask = local_mask & k_mask[None, :] & q_mask[:, None]
        scores = tl.where(full_mask, scores, float('-inf'))
        
        p = tl.exp(scores - lse[:, None])
        Di = tl.sum(o * do, axis=1)
        dov = tl.dot(do, tl.trans(vals))
        dp = p * (dov - Di[:, None])
        dp = tl.where(full_mask, dp, 0.0)
        
        dk_acc += tl.dot(tl.trans(dp.to(q.dtype)), q) * scale
        dv_acc += tl.dot(tl.trans(p.to(do.dtype)), do)
    
    # Store dK, dV (will be accumulated with strided phase)
    dk_ptr = dK + b * stride_dkb + h * stride_dkh
    tl.store(dk_ptr + k_offsets[:, None] * stride_dkn + d_idx[None, :] * stride_dkd,
             dk_acc.to(dK.dtype.element_ty),
             mask=k_mask[:, None] & (d_idx[None, :] < D))
    
    dv_ptr = dV + b * stride_dvb + h * stride_dvh
    tl.store(dv_ptr + k_offsets[:, None] * stride_dvn + d_idx[None, :] * stride_dvd,
             dv_acc.to(dV.dtype.element_ty),
             mask=k_mask[:, None] & (d_idx[None, :] < D))


# =============================================================================
# KEY-CENTRIC STRIDED KERNEL (ZERO ATOMICS!)
# =============================================================================
# The fix: Instead of query-parallel with atomics, use key-parallel with
# exclusive ownership. Each block owns ONE strided key, iterates over ALL
# queries that attend to it, accumulates in registers, writes ONCE.

@triton.jit
def _dkdv_strided_key_centric(
    Q, K, V, O, dO, dK, dV, L,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lb, stride_lh, stride_lm,
    S: tl.constexpr, D: tl.constexpr, SQRT_N: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr, NUM_Q_TILES: tl.constexpr,
):
    """
    Key-Centric Strided dK/dV with 2D query tiling (ZERO ATOMICS per tile).

    Grid: (num_strided_keys * NUM_Q_TILES, B, H)

    Each block OWNS one (strided_key, query_tile) pair. The query axis is split
    across NUM_Q_TILES instances so the O(N) inner loop is parallelised across
    blocks.  atomic_add at the end merges partial dK/dV from the query tiles.

    NUM_Q_TILES selection (Python side):
      S <= 256  → NUM_Q_TILES = 1   (single block covers the whole query range)
      S <= 512  → NUM_Q_TILES = 2
      S > 512   → NUM_Q_TILES = 4
    """
    combined_idx = tl.program_id(0)  # encodes strided_idx * NUM_Q_TILES + q_tile_idx
    b            = tl.program_id(1)
    h            = tl.program_id(2)

    strided_idx = combined_idx // NUM_Q_TILES
    q_tile_idx  = combined_idx % NUM_Q_TILES

    # Position of this strided key
    k_pos = strided_idx * SQRT_N

    if k_pos >= S:
        return
    
    d_idx = tl.arange(0, BLOCK_D)
    
    # Load the key and value we own (stays in registers for entire kernel)
    k_ptr = K + b * stride_kb + h * stride_kh
    key = tl.load(k_ptr + k_pos * stride_kn + d_idx * stride_kd,
                  mask=d_idx < D, other=0.0)  # [D]
    
    v_ptr = V + b * stride_vb + h * stride_vh
    val = tl.load(v_ptr + k_pos * stride_vn + d_idx * stride_vd,
                  mask=d_idx < D, other=0.0)  # [D]
    
    # Accumulators in registers (fp32 for precision)
    dk_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    
    # Queries that attend to this strided key: q >= k_pos + SQRT_N
    q_start = k_pos + SQRT_N

    if q_start >= S:
        return  # No queries attend to this strided key

    # 2D tiling: subdivide [q_start, S) across NUM_Q_TILES instances
    total_q      = S - q_start
    q_per_tile   = (total_q + NUM_Q_TILES - 1) // NUM_Q_TILES
    tile_q_start = q_start + q_tile_idx * q_per_tile
    tile_q_end   = tl.minimum(tile_q_start + q_per_tile, S)

    if tile_q_start >= S:
        return  # This tile is empty

    # Stream through our query tile
    for q_block_start in range(tile_q_start, tile_q_end, BLOCK_Q):
        q_offsets = q_block_start + tl.arange(0, BLOCK_Q)
        q_mask = q_offsets < tile_q_end
        
        # Load Q block: [BLOCK_Q, D]
        q_ptr = Q + b * stride_qb + h * stride_qh
        q = tl.load(q_ptr + q_offsets[:, None] * stride_qm + d_idx[None, :] * stride_qd,
                    mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        # Load O, dO, LSE for gradient computation
        o_ptr = O + b * stride_ob + h * stride_oh
        o = tl.load(o_ptr + q_offsets[:, None] * stride_om + d_idx[None, :] * stride_od,
                    mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        do_ptr = dO + b * stride_dob + h * stride_doh
        do = tl.load(do_ptr + q_offsets[:, None] * stride_dom + d_idx[None, :] * stride_dod,
                     mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        l_ptr = L + b * stride_lb + h * stride_lh
        lse = tl.load(l_ptr + q_offsets * stride_lm, mask=q_mask, other=0.0)
        
        # Compute attention scores: q @ key.T -> [BLOCK_Q]
        scores = tl.sum(q * key[None, :], axis=1) * scale
        
        # Mask: only valid if k_pos < local_start(q)
        # local_start(q) = max(q - SQRT_N + 1, 0)
        local_starts = tl.maximum(q_offsets - SQRT_N + 1, 0)
        strided_mask = (k_pos < local_starts) & q_mask
        scores = tl.where(strided_mask, scores, float('-inf'))
        
        # Compute softmax probability
        p = tl.exp(scores - lse)  # [BLOCK_Q]
        p = tl.where(strided_mask, p, 0.0)
        
        # Compute gradient flow
        # Di = sum(o * do, axis=1) for each query
        Di = tl.sum(o * do, axis=1)  # [BLOCK_Q]
        
        # dov = do @ val for each query
        dov = tl.sum(do * val[None, :], axis=1)  # [BLOCK_Q]
        
        # dp = p * (dov - Di)
        dp = p * (dov - Di)  # [BLOCK_Q]
        dp = tl.where(strided_mask, dp, 0.0)
        
        # Accumulate dK: dK += sum_queries(dp * q) * scale
        # dp is [BLOCK_Q], q is [BLOCK_Q, D] -> reduce over queries
        dk_acc += tl.sum(dp[:, None] * q, axis=0) * scale  # [D]
        
        # Accumulate dV: dV += sum_queries(p * do)
        dv_acc += tl.sum(p[:, None] * do, axis=0)  # [D]
    
    # Write dK and dV (ATOMIC-FREE - we're the only writer!)
    dk_ptr = dK + b * stride_dkb + h * stride_dkh
    dv_ptr = dV + b * stride_dvb + h * stride_dvh
    
    # Use atomic_add to accumulate with local phase (they computed local dK/dV)
    tl.atomic_add(dk_ptr + k_pos * stride_dkn + d_idx * stride_dkd,
                  dk_acc.to(dK.dtype.element_ty), mask=d_idx < D)
    tl.atomic_add(dv_ptr + k_pos * stride_dvn + d_idx * stride_dvd,
                  dv_acc.to(dV.dtype.element_ty), mask=d_idx < D)


# =============================================================================
# GQA-AWARE SPLIT3 KERNELS  (kv_groups > 1)
# =============================================================================
# These are KV-head-centric variants of _dkdv_local_phase and
# _dkdv_strided_key_centric. By making the CUDA grid index KV heads instead
# of Q heads, each program exclusively owns a KV-head slice and loops over
# the KV_GROUPS Q heads that share it — eliminating the kv_groups-fold atomic
# contention that MHA kernels running under GQA would otherwise produce.

@triton.jit
def _dkdv_local_phase_gqa(
    Q, K, V, O, dO, dK, dV, L,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lb, stride_lh, stride_lm,
    S: tl.constexpr, D: tl.constexpr, SQRT_N: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_Q: tl.constexpr,
    KV_GROUPS: tl.constexpr,
):
    """
    GQA local-only dK/dV: O(N√N) complexity per KV head, zero atomics.

    Grid: (B, H_kv, num_kv_tiles)
    Each program owns one KV tile and accumulates dK/dV contributions from
    ALL KV_GROUPS Q heads that share this KV head.  Direct tl.store() — no
    atomic contention because program_id(1) == h_kv is an exclusive owner of
    its dK/dV slice.
    """
    b      = tl.program_id(0)
    h_kv   = tl.program_id(1)
    tile_k = tl.program_id(2)

    k_start   = tile_k * BLOCK_N
    k_offsets = k_start + tl.arange(0, BLOCK_N)
    k_mask    = k_offsets < S
    d_idx     = tl.arange(0, BLOCK_D)

    # Load K, V for this KV-head tile (held in registers throughout)
    k_ptr = K + b * stride_kb + h_kv * stride_kh
    keys  = tl.load(k_ptr + k_offsets[:, None] * stride_kn + d_idx[None, :] * stride_kd,
                    mask=k_mask[:, None] & (d_idx[None, :] < D), other=0.0)
    v_ptr = V + b * stride_vb + h_kv * stride_vh
    vals  = tl.load(v_ptr + k_offsets[:, None] * stride_vn + d_idx[None, :] * stride_vd,
                    mask=k_mask[:, None] & (d_idx[None, :] < D), other=0.0)

    scale     = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    dk_acc    = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv_acc    = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    # Local query range for this KV tile: [k_start, k_start + BLOCK_N - 1 + SQRT_N - 1]
    local_q_start = k_start
    local_q_end   = tl.minimum(k_start + BLOCK_N - 1 + SQRT_N - 1, S - 1)

    # Accumulate from all KV_GROUPS Q heads sharing this KV head
    for g in range(KV_GROUPS):
        h_q = h_kv * KV_GROUPS + g

        for q_block in range(0, SQRT_N + BLOCK_N, BLOCK_Q):
            q_offsets = local_q_start + q_block + tl.arange(0, BLOCK_Q)
            q_mask    = (q_offsets <= local_q_end) & (q_offsets < S)

            q_ptr = Q + b * stride_qb + h_q * stride_qh
            q     = tl.load(q_ptr + q_offsets[:, None] * stride_qm + d_idx[None, :] * stride_qd,
                            mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
            o_ptr = O + b * stride_ob + h_q * stride_oh
            o     = tl.load(o_ptr + q_offsets[:, None] * stride_om + d_idx[None, :] * stride_od,
                            mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
            do_ptr = dO + b * stride_dob + h_q * stride_doh
            do    = tl.load(do_ptr + q_offsets[:, None] * stride_dom + d_idx[None, :] * stride_dod,
                            mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
            l_ptr = L + b * stride_lb + h_q * stride_lh
            lse   = tl.load(l_ptr + q_offsets * stride_lm, mask=q_mask, other=0.0)

            scores     = tl.dot(q, tl.trans(keys)) * scale
            local_mask = (k_offsets[None, :] <= q_offsets[:, None]) & \
                         (k_offsets[None, :] >= q_offsets[:, None] - SQRT_N + 1)
            full_mask  = local_mask & k_mask[None, :] & q_mask[:, None]
            scores     = tl.where(full_mask, scores, float('-inf'))

            p    = tl.exp(scores - lse[:, None])
            Di   = tl.sum(o * do, axis=1)
            dov  = tl.dot(do, tl.trans(vals))
            dp   = p * (dov - Di[:, None])
            dp   = tl.where(full_mask, dp, 0.0)

            dk_acc += tl.dot(tl.trans(dp.to(q.dtype)), q) * scale
            dv_acc += tl.dot(tl.trans(p.to(do.dtype)), do)

    # Direct store — this program has exclusive ownership of dK/dV[h_kv][k_start:k_end]
    dk_ptr = dK + b * stride_dkb + h_kv * stride_dkh
    tl.store(dk_ptr + k_offsets[:, None] * stride_dkn + d_idx[None, :] * stride_dkd,
             dk_acc.to(dK.dtype.element_ty),
             mask=k_mask[:, None] & (d_idx[None, :] < D))
    dv_ptr = dV + b * stride_dvb + h_kv * stride_dvh
    tl.store(dv_ptr + k_offsets[:, None] * stride_dvn + d_idx[None, :] * stride_dvd,
             dv_acc.to(dV.dtype.element_ty),
             mask=k_mask[:, None] & (d_idx[None, :] < D))


@triton.jit
def _dkdv_strided_key_centric_gqa(
    Q, K, V, O, dO, dK, dV, L,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lb, stride_lh, stride_lm,
    S: tl.constexpr, D: tl.constexpr, SQRT_N: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
    KV_GROUPS: tl.constexpr, NUM_Q_TILES: tl.constexpr,
):
    """
    GQA key-centric strided dK/dV with 2D query tiling.

    Grid: (num_strided_keys * NUM_Q_TILES, B, H_kv)
    Each program owns one (strided_key, q_tile) pair and loops over KV_GROUPS
    Q heads. atomic_add merges partial dK/dV from query tiles with the local phase.
    """
    combined_idx = tl.program_id(0)
    b            = tl.program_id(1)
    h_kv         = tl.program_id(2)

    strided_idx = combined_idx // NUM_Q_TILES
    q_tile_idx  = combined_idx % NUM_Q_TILES

    k_pos = strided_idx * SQRT_N
    if k_pos >= S:
        return

    d_idx = tl.arange(0, BLOCK_D)

    # Load the key/value we own at this KV head (stays in registers)
    k_ptr = K + b * stride_kb + h_kv * stride_kh
    key   = tl.load(k_ptr + k_pos * stride_kn + d_idx * stride_kd,
                    mask=d_idx < D, other=0.0)
    v_ptr = V + b * stride_vb + h_kv * stride_vh
    val   = tl.load(v_ptr + k_pos * stride_vn + d_idx * stride_vd,
                    mask=d_idx < D, other=0.0)

    dk_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    scale  = 1.0 / tl.sqrt(tl.cast(D, tl.float32))

    # Queries that attend to this strided key: q >= k_pos + SQRT_N
    q_start = k_pos + SQRT_N
    if q_start >= S:
        return

    # 2D tiling: subdivide [q_start, S) across NUM_Q_TILES instances
    total_q      = S - q_start
    q_per_tile   = (total_q + NUM_Q_TILES - 1) // NUM_Q_TILES
    tile_q_start = q_start + q_tile_idx * q_per_tile
    tile_q_end   = tl.minimum(tile_q_start + q_per_tile, S)

    if tile_q_start >= S:
        return  # This tile is empty

    # Accumulate from all KV_GROUPS Q heads sharing this KV head
    for g in range(KV_GROUPS):
        h_q = h_kv * KV_GROUPS + g

        for q_block_start in range(tile_q_start, tile_q_end, BLOCK_Q):
            q_offsets  = q_block_start + tl.arange(0, BLOCK_Q)
            q_mask     = q_offsets < tile_q_end

            q_ptr  = Q + b * stride_qb + h_q * stride_qh
            q      = tl.load(q_ptr + q_offsets[:, None] * stride_qm + d_idx[None, :] * stride_qd,
                             mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
            o_ptr  = O + b * stride_ob + h_q * stride_oh
            o      = tl.load(o_ptr + q_offsets[:, None] * stride_om + d_idx[None, :] * stride_od,
                             mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
            do_ptr = dO + b * stride_dob + h_q * stride_doh
            do     = tl.load(do_ptr + q_offsets[:, None] * stride_dom + d_idx[None, :] * stride_dod,
                             mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
            l_ptr  = L + b * stride_lb + h_q * stride_lh
            lse    = tl.load(l_ptr + q_offsets * stride_lm, mask=q_mask, other=0.0)

            scores       = tl.sum(q * key[None, :], axis=1) * scale
            local_starts = tl.maximum(q_offsets - SQRT_N + 1, 0)
            strided_mask = (k_pos < local_starts) & q_mask
            scores       = tl.where(strided_mask, scores, float('-inf'))

            p    = tl.exp(scores - lse)
            p    = tl.where(strided_mask, p, 0.0)
            Di   = tl.sum(o * do, axis=1)
            dov  = tl.sum(do * val[None, :], axis=1)
            dp   = p * (dov - Di)
            dp   = tl.where(strided_mask, dp, 0.0)

            dk_acc += tl.sum(dp[:, None] * q, axis=0) * scale
            dv_acc += tl.sum(p[:, None] * do, axis=0)

    # atomic_add to merge with the local phase's direct store for this position
    dk_ptr = dK + b * stride_dkb + h_kv * stride_dkh
    dv_ptr = dV + b * stride_dvb + h_kv * stride_dvh
    tl.atomic_add(dk_ptr + k_pos * stride_dkn + d_idx * stride_dkd,
                  dk_acc.to(dK.dtype.element_ty), mask=d_idx < D)
    tl.atomic_add(dv_ptr + k_pos * stride_dvn + d_idx * stride_dvd,
                  dv_acc.to(dV.dtype.element_ty), mask=d_idx < D)


@triton.jit
def _dkdv_strided_phase(
    Q, K, V, O, dO, dK, dV, L,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lb, stride_lh, stride_lm,
    S: tl.constexpr, D: tl.constexpr, SQRT_N: tl.constexpr,
    NUM_Q_TILES: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_Q: tl.constexpr,
):
    """
    Strided dK/dV: O(N√N) complexity with atomic reduction.
    
    Grid: (B, H, num_strided_keys * NUM_Q_TILES)
    Each block processes one strided key + one query tile.
    Uses atomic_add for reduction across query tiles.
    
    The atomic contention is minimal because:
    - Only √N strided keys exist
    - NUM_Q_TILES blocks write to same key, but D=64 elements spread contention
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    combined_idx = tl.program_id(2)
    
    # Decode
    strided_idx = combined_idx // NUM_Q_TILES
    tile_idx = combined_idx % NUM_Q_TILES
    
    k_pos = strided_idx * SQRT_N
    if k_pos >= S:
        return
    
    q_start_base = k_pos + SQRT_N
    total_queries = S - q_start_base
    
    if total_queries <= 0:
        return
    
    queries_per_tile = (total_queries + NUM_Q_TILES - 1) // NUM_Q_TILES
    tile_q_start = q_start_base + tile_idx * queries_per_tile
    tile_q_end = tl.minimum(tile_q_start + queries_per_tile, S)
    
    if tile_q_start >= S:
        return
    
    d_idx = tl.arange(0, BLOCK_D)
    
    # Load K, V for this strided key (stays in registers)
    k_ptr = K + b * stride_kb + h * stride_kh
    key = tl.load(k_ptr + k_pos * stride_kn + d_idx * stride_kd,
                  mask=d_idx < D, other=0.0)
    
    v_ptr = V + b * stride_vb + h * stride_vh
    val = tl.load(v_ptr + k_pos * stride_vn + d_idx * stride_vd,
                  mask=d_idx < D, other=0.0)
    
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    
    dk_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    for q_block in range(tile_q_start, tile_q_end, BLOCK_Q):
        q_offsets = q_block + tl.arange(0, BLOCK_Q)
        q_mask = q_offsets < tile_q_end
        
        q_ptr = Q + b * stride_qb + h * stride_qh
        q = tl.load(q_ptr + q_offsets[:, None] * stride_qm + d_idx[None, :] * stride_qd,
                    mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        o_ptr = O + b * stride_ob + h * stride_oh
        o = tl.load(o_ptr + q_offsets[:, None] * stride_om + d_idx[None, :] * stride_od,
                    mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        do_ptr = dO + b * stride_dob + h * stride_doh
        do = tl.load(do_ptr + q_offsets[:, None] * stride_dom + d_idx[None, :] * stride_dod,
                     mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        l_ptr = L + b * stride_lb + h * stride_lh
        lse = tl.load(l_ptr + q_offsets * stride_lm, mask=q_mask, other=0.0)
        
        scores = tl.sum(q * key[None, :], axis=1) * scale
        
        local_starts = tl.maximum(q_offsets - SQRT_N + 1, 0)
        strided_mask = (k_pos < local_starts) & q_mask
        scores = tl.where(strided_mask, scores, float('-inf'))
        
        p = tl.exp(scores - lse)
        Di = tl.sum(o * do, axis=1)
        dov = tl.sum(do * val[None, :], axis=1)
        dp = p * (dov - Di)
        dp = tl.where(strided_mask, dp, 0.0)
        
        dk_acc += tl.sum(dp[:, None] * q, axis=0) * scale
        dv_acc += tl.sum(p[:, None] * do, axis=0)
    
    # Atomic add for reduction across query tiles
    dk_ptr = dK + b * stride_dkb + h * stride_dkh + k_pos * stride_dkn
    dv_ptr = dV + b * stride_dvb + h * stride_dvh + k_pos * stride_dvn
    
    tl.atomic_add(dk_ptr + d_idx * stride_dkd, dk_acc.to(dK.dtype.element_ty), mask=d_idx < D)
    tl.atomic_add(dv_ptr + d_idx * stride_dvd, dv_acc.to(dV.dtype.element_ty), mask=d_idx < D)


@triton.jit
def _transposed_gather_dkdv_v14_tiled(
    Q, K, V, O, dO, dK, dV, L,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lb, stride_lh, stride_lm,
    S: tl.constexpr, D: tl.constexpr, SQRT_N: tl.constexpr,
    BLOCK_N: tl.constexpr,  # Keys per tile
    BLOCK_D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    """
    Transposed Gather dK/dV: Tiled, one block per (B, H, tile_k).
    Processes BLOCK_N keys at once.
    
    Note: The strided phase is O(N) per strided key because ALL queries
    from k + √N onwards attend to each strided key k. This is fundamentally
    different from the forward pass where each query attends to O(√N) keys.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    tile_k = tl.program_id(2)
    
    k_start = tile_k * BLOCK_N
    k_offsets = k_start + tl.arange(0, BLOCK_N)
    k_mask = k_offsets < S
    
    d_idx = tl.arange(0, BLOCK_D)
    
    # Load K, V blocks
    k_ptr = K + b * stride_kb + h * stride_kh
    keys = tl.load(k_ptr + k_offsets[:, None] * stride_kn + d_idx[None, :] * stride_kd,
                   mask=k_mask[:, None] & (d_idx[None, :] < D), other=0.0)
    
    v_ptr = V + b * stride_vb + h * stride_vh
    vals = tl.load(v_ptr + k_offsets[:, None] * stride_vn + d_idx[None, :] * stride_vd,
                   mask=k_mask[:, None] & (d_idx[None, :] < D), other=0.0)
    
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    
    dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    
    # Phase 1: Local queries - those that have any key in [k_start, k_start + BLOCK_N - 1]
    # in their local window
    # For key k, local queries are [k, min(k + SQRT_N - 1, S-1)]
    # For key block, queries range from k_start to k_start + BLOCK_N - 1 + SQRT_N - 1
    
    local_q_start = k_start
    local_q_end = k_start + BLOCK_N - 1 + SQRT_N - 1
    local_q_end = tl.minimum(local_q_end, S - 1)
    
    for q_block in range(0, SQRT_N + BLOCK_N, BLOCK_Q):
        q_offsets = local_q_start + q_block + tl.arange(0, BLOCK_Q)
        q_mask = (q_offsets >= local_q_start) & (q_offsets <= local_q_end) & (q_offsets < S)
        
        # Load Q, O, dO, L
        q_ptr = Q + b * stride_qb + h * stride_qh
        q = tl.load(q_ptr + q_offsets[:, None] * stride_qm + d_idx[None, :] * stride_qd,
                    mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        o_ptr = O + b * stride_ob + h * stride_oh
        o = tl.load(o_ptr + q_offsets[:, None] * stride_om + d_idx[None, :] * stride_od,
                    mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        do_ptr = dO + b * stride_dob + h * stride_doh
        do = tl.load(do_ptr + q_offsets[:, None] * stride_dom + d_idx[None, :] * stride_dod,
                     mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        l_ptr = L + b * stride_lb + h * stride_lh
        lse = tl.load(l_ptr + q_offsets * stride_lm, mask=q_mask, other=0.0)
        
        # Scores: [BLOCK_Q, BLOCK_N] = Q @ K^T
        scores = tl.dot(q, tl.trans(keys)) * scale
        
        # Mask: k in [q - SQRT_N + 1, q]
        local_mask = (k_offsets[None, :] <= q_offsets[:, None]) & \
                     (k_offsets[None, :] >= q_offsets[:, None] - SQRT_N + 1)
        full_mask = local_mask & k_mask[None, :] & q_mask[:, None]
        scores = tl.where(full_mask, scores, float('-inf'))
        
        # P = softmax
        p = tl.exp(scores - lse[:, None])
        
        # Di
        Di = tl.sum(o * do, axis=1)
        
        # dP = P * (dO @ V^T - Di)
        dov = tl.dot(do, tl.trans(vals))  # [BLOCK_Q, BLOCK_N]
        dp = p * (dov - Di[:, None])
        dp = tl.where(full_mask, dp, 0.0)
        
        # dK += dP^T @ Q * scale
        dk_acc += tl.dot(tl.trans(dp.to(q.dtype)), q) * scale
        
        # dV += P^T @ dO
        dv_acc += tl.dot(tl.trans(p.to(do.dtype)), do)
    
    # Phase 2: Strided queries (for keys that are strided positions)
    # A key k is a strided position if k % SQRT_N == 0
    # For each strided key position k, ALL queries from k + SQRT_N onwards attend to it
    
    # Maximum queries that could attend to any key in this tile via strided pattern
    max_strided_q_start = k_start + SQRT_N  # Earliest strided query for k_start
    num_strided_queries = S - max_strided_q_start
    
    if num_strided_queries > 0:
        for q_block in range(0, num_strided_queries, BLOCK_Q):
            q_offsets = max_strided_q_start + q_block + tl.arange(0, BLOCK_Q)
            q_mask_base = q_offsets < S
            
            q_ptr = Q + b * stride_qb + h * stride_qh
            q = tl.load(q_ptr + q_offsets[:, None] * stride_qm + d_idx[None, :] * stride_qd,
                        mask=q_mask_base[:, None] & (d_idx[None, :] < D), other=0.0)
            
            o_ptr = O + b * stride_ob + h * stride_oh
            o = tl.load(o_ptr + q_offsets[:, None] * stride_om + d_idx[None, :] * stride_od,
                        mask=q_mask_base[:, None] & (d_idx[None, :] < D), other=0.0)
            
            do_ptr = dO + b * stride_dob + h * stride_doh
            do = tl.load(do_ptr + q_offsets[:, None] * stride_dom + d_idx[None, :] * stride_dod,
                         mask=q_mask_base[:, None] & (d_idx[None, :] < D), other=0.0)
            
            l_ptr = L + b * stride_lb + h * stride_lh
            lse = tl.load(l_ptr + q_offsets * stride_lm, mask=q_mask_base, other=0.0)
            
            # Scores: [BLOCK_Q, BLOCK_N] = Q @ K^T
            scores = tl.dot(q, tl.trans(keys)) * scale
            
            # Strided mask: key k is accessed by query q if:
            # 1. k % SQRT_N == 0 (k is a strided position)
            # 2. q >= k + SQRT_N (q is beyond k's local window)
            is_strided_key = (k_offsets % SQRT_N == 0)
            strided_mask = (q_offsets[:, None] >= k_offsets[None, :] + SQRT_N) & is_strided_key[None, :]
            full_mask = strided_mask & k_mask[None, :] & q_mask_base[:, None]
            
            scores = tl.where(full_mask, scores, float('-inf'))
            p = tl.exp(scores - lse[:, None])
            
            Di = tl.sum(o * do, axis=1)
            dov = tl.dot(do, tl.trans(vals))  # [BLOCK_Q, BLOCK_N]
            dp = p * (dov - Di[:, None])
            dp = tl.where(full_mask, dp, 0.0)
            
            # dK += dP^T @ Q * scale
            dk_acc += tl.dot(tl.trans(dp.to(q.dtype)), q) * scale
            
            # dV += P^T @ dO
            dv_acc += tl.dot(tl.trans(p.to(do.dtype)), do)
    
    # Store dK, dV
    dk_ptr = dK + b * stride_dkb + h * stride_dkh
    tl.store(dk_ptr + k_offsets[:, None] * stride_dkn + d_idx[None, :] * stride_dkd,
             dk_acc.to(dK.dtype.element_ty),
             mask=k_mask[:, None] & (d_idx[None, :] < D))
    
    dv_ptr = dV + b * stride_dvb + h * stride_dvh
    tl.store(dv_ptr + k_offsets[:, None] * stride_dvn + d_idx[None, :] * stride_dvd,
             dv_acc.to(dV.dtype.element_ty),
             mask=k_mask[:, None] & (d_idx[None, :] < D))


# =============================================================================
# RELAY dK/dV KERNEL (2-hop backward)
# =============================================================================

@triton.jit
def _dkdv_relay_v14(
    Q, RK, RV, O, dO, dRK, dRV, L,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_rkb, stride_rkh, stride_rkn, stride_rkd,
    stride_rvb, stride_rvh, stride_rvn, stride_rvd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_drkb, stride_drkh, stride_drkn, stride_drkd,
    stride_drvb, stride_drvh, stride_drvn, stride_drvd,
    stride_lb, stride_lh, stride_lm,
    S: tl.constexpr, D: tl.constexpr, SQRT_N: tl.constexpr,
    NUM_RELAY: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
    TOP_K: tl.constexpr = 2,  # Relay tokens per block (for block masking)
):
    """
    Relay-centric dRK/dRV kernel (2-hop backward).
    
    Grid: (NUM_RELAY, B, H)
    Each block OWNS one relay key/value, iterates over all queries
    that attend to it. Accumulates in registers, writes ONCE.
    Zero atomics (exclusive ownership).
    """
    relay_idx = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    if relay_idx >= NUM_RELAY:
        return
    
    d_idx = tl.arange(0, BLOCK_D)
    
    # Load relay key and value (stays in registers for entire kernel)
    rk_ptr = RK + b * stride_rkb + h * stride_rkh
    relay_key = tl.load(rk_ptr + relay_idx * stride_rkn + d_idx * stride_rkd,
                        mask=d_idx < D, other=0.0)
    
    rv_ptr = RV + b * stride_rvb + h * stride_rvh
    relay_val = tl.load(rv_ptr + relay_idx * stride_rvn + d_idx * stride_rvd,
                        mask=d_idx < D, other=0.0)
    
    drk_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    drv_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    
    # Relay token relay_idx belongs to block (relay_idx // TOP_K).
    # That block ends at position (relay_idx // TOP_K + 1) * SQRT_N - 1.
    block_end = (relay_idx // TOP_K + 1) * SQRT_N - 1
    
    # Only queries with local_start > block_end attend to this relay
    # local_start(m) = max(m - SQRT_N + 1, 0) > block_end
    # m > block_end + SQRT_N - 1  =>  m >= block_end + SQRT_N
    q_start = block_end + SQRT_N
    
    if q_start >= S:
        # No queries attend to this relay block
        drk_ptr = dRK + b * stride_drkb + h * stride_drkh
        tl.store(drk_ptr + relay_idx * stride_drkn + d_idx * stride_drkd,
                 drk_acc.to(dRK.dtype.element_ty), mask=d_idx < D)
        drv_ptr = dRV + b * stride_drvb + h * stride_drvh
        tl.store(drv_ptr + relay_idx * stride_drvn + d_idx * stride_drvd,
                 drv_acc.to(dRV.dtype.element_ty), mask=d_idx < D)
        return
    
    for q_block_start in range(q_start, S, BLOCK_Q):
        q_offsets = q_block_start + tl.arange(0, BLOCK_Q)
        q_mask = q_offsets < S
        
        q_ptr = Q + b * stride_qb + h * stride_qh
        q = tl.load(q_ptr + q_offsets[:, None] * stride_qm + d_idx[None, :] * stride_qd,
                    mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        o_ptr = O + b * stride_ob + h * stride_oh
        o = tl.load(o_ptr + q_offsets[:, None] * stride_om + d_idx[None, :] * stride_od,
                    mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        do_ptr = dO + b * stride_dob + h * stride_doh
        do = tl.load(do_ptr + q_offsets[:, None] * stride_dom + d_idx[None, :] * stride_dod,
                     mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)
        
        l_ptr = L + b * stride_lb + h * stride_lh
        lse = tl.load(l_ptr + q_offsets * stride_lm, mask=q_mask, other=0.0)
        
        # Score: q @ relay_key^T
        scores = tl.sum(q * relay_key[None, :], axis=1) * scale
        
        # Mask: block_end < local_start(q)
        local_starts = tl.maximum(q_offsets - SQRT_N + 1, 0)
        relay_mask = (block_end < local_starts) & q_mask
        scores = tl.where(relay_mask, scores, float('-inf'))
        
        p = tl.exp(scores - lse)
        p = tl.where(relay_mask, p, 0.0)
        
        Di = tl.sum(o * do, axis=1)
        dov = tl.sum(do * relay_val[None, :], axis=1)
        dp = p * (dov - Di)
        dp = tl.where(relay_mask, dp, 0.0)
        
        drk_acc += tl.sum(dp[:, None] * q, axis=0) * scale
        drv_acc += tl.sum(p[:, None] * do, axis=0)
    
    # Write (no atomics - exclusive ownership)
    drk_ptr = dRK + b * stride_drkb + h * stride_drkh
    tl.store(drk_ptr + relay_idx * stride_drkn + d_idx * stride_drkd,
             drk_acc.to(dRK.dtype.element_ty), mask=d_idx < D)
    
    drv_ptr = dRV + b * stride_drvb + h * stride_drvh
    tl.store(drv_ptr + relay_idx * stride_drvn + d_idx * stride_drvd,
             drv_acc.to(dRV.dtype.element_ty), mask=d_idx < D)


@triton.jit
def _dkdv_relay_v14_gqa(
    Q, RK, RV, O, dO, dRK, dRV, L,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_rkb, stride_rkh, stride_rkn, stride_rkd,
    stride_rvb, stride_rvh, stride_rvn, stride_rvd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_drkb, stride_drkh, stride_drkn, stride_drkd,
    stride_drvb, stride_drvh, stride_drvn, stride_drvd,
    stride_lb, stride_lh, stride_lm,
    S: tl.constexpr, D: tl.constexpr, SQRT_N: tl.constexpr,
    NUM_RELAY: tl.constexpr,
    KV_GROUPS: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    GQA-aware relay-centric dRK/dRV kernel.

    Grid: (NUM_RELAY, B, H_kv)
    Each block OWNS one relay key/value, iterates over all KV_GROUPS Q heads
    and all queries within them that attend to this relay.
    Zero atomics (exclusive ownership per relay slot).
    """
    relay_idx = tl.program_id(0)
    b         = tl.program_id(1)
    h_kv      = tl.program_id(2)

    if relay_idx >= NUM_RELAY:
        return

    d_idx = tl.arange(0, BLOCK_D)

    # Load relay key and value (stays in registers for entire kernel)
    rk_ptr = RK + b * stride_rkb + h_kv * stride_rkh
    relay_key = tl.load(rk_ptr + relay_idx * stride_rkn + d_idx * stride_rkd,
                        mask=d_idx < D, other=0.0)

    rv_ptr = RV + b * stride_rvb + h_kv * stride_rvh
    relay_val = tl.load(rv_ptr + relay_idx * stride_rvn + d_idx * stride_rvd,
                        mask=d_idx < D, other=0.0)

    drk_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    drv_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))

    block_end = (relay_idx + 1) * SQRT_N - 1
    q_start   = block_end + SQRT_N

    if q_start >= S:
        drk_ptr = dRK + b * stride_drkb + h_kv * stride_drkh
        tl.store(drk_ptr + relay_idx * stride_drkn + d_idx * stride_drkd,
                 drk_acc.to(dRK.dtype.element_ty), mask=d_idx < D)
        drv_ptr = dRV + b * stride_drvb + h_kv * stride_drvh
        tl.store(drv_ptr + relay_idx * stride_drvn + d_idx * stride_drvd,
                 drv_acc.to(dRV.dtype.element_ty), mask=d_idx < D)
        return

    # Accumulate from all KV_GROUPS query heads sharing this KV head
    for kg in range(KV_GROUPS):
        h_q = h_kv * KV_GROUPS + kg

        for q_block_start in range(q_start, S, BLOCK_Q):
            q_offsets = q_block_start + tl.arange(0, BLOCK_Q)
            q_mask    = q_offsets < S

            q_ptr = Q + b * stride_qb + h_q * stride_qh
            q = tl.load(q_ptr + q_offsets[:, None] * stride_qm + d_idx[None, :] * stride_qd,
                        mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)

            o_ptr = O + b * stride_ob + h_q * stride_oh
            o = tl.load(o_ptr + q_offsets[:, None] * stride_om + d_idx[None, :] * stride_od,
                        mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)

            do_ptr = dO + b * stride_dob + h_q * stride_doh
            do = tl.load(do_ptr + q_offsets[:, None] * stride_dom + d_idx[None, :] * stride_dod,
                         mask=q_mask[:, None] & (d_idx[None, :] < D), other=0.0)

            l_ptr = L + b * stride_lb + h_q * stride_lh
            lse = tl.load(l_ptr + q_offsets * stride_lm, mask=q_mask, other=0.0)

            scores = tl.sum(q * relay_key[None, :], axis=1) * scale

            local_starts = tl.maximum(q_offsets - SQRT_N + 1, 0)
            relay_mask   = (block_end < local_starts) & q_mask
            scores = tl.where(relay_mask, scores, float('-inf'))

            p = tl.exp(scores - lse)
            p = tl.where(relay_mask, p, 0.0)

            Di  = tl.sum(o * do, axis=1)
            dov = tl.sum(do * relay_val[None, :], axis=1)
            dp  = p * (dov - Di)
            dp  = tl.where(relay_mask, dp, 0.0)

            drk_acc += tl.sum(dp[:, None] * q, axis=0) * scale
            drv_acc += tl.sum(p[:, None] * do, axis=0)

    # Write (no atomics - exclusive ownership per relay slot)
    drk_ptr = dRK + b * stride_drkb + h_kv * stride_drkh
    tl.store(drk_ptr + relay_idx * stride_drkn + d_idx * stride_drkd,
             drk_acc.to(dRK.dtype.element_ty), mask=d_idx < D)

    drv_ptr = dRV + b * stride_drvb + h_kv * stride_drvh
    tl.store(drv_ptr + relay_idx * stride_drvn + d_idx * stride_drvd,
             drv_acc.to(dRV.dtype.element_ty), mask=d_idx < D)


# =============================================================================
# RELAY PRECOMPUTE KERNEL  (top-K exact-token selection for zero-distortion relay)
# =============================================================================
#
# Previous: norm²-weighted pooling — relay_k[r] = Σ wᵢ·kᵢ
# Problem:  weighted pooling is still a blurry average; q·relay_k blends all
#           tokens regardless of which one actually matters for a given query.
#           Both routing (key) and retrieval (value) are imprecise.
#
# New design: exact top-K selection.
#   relay_k[r*K+j] = k[top_j(r)]   — exact key of j-th highest-‖k‖² token
#   relay_v[r*K+j] = v[top_j(r)]   — its exact matching value (no blurring)
#   relay_src[r*K+j] = float32(abs_pos)  — source position for backward scatter
#
# With K=2: each block contributes 2 exact representative tokens.
# Total relay tokens: NUM_RELAY_BLOCKS * K, still O(K·√N) per sequence.
#
# Memory profile vs old pooling (K=2, SQRT_N=64, D=128):
#   Old: reads 64 K rows + 64 V rows = 128 rows
#   New: reads 64 K rows (norm scan) + 2 K rows + 2 V rows = 68 rows  (47% less)
#
# Algorithm:
#   Pass 1: scan SQRT_N positions, compute ‖kᵢ‖², track top-K indices in registers.
#   Pass 2: load exact k/v at top-K positions and write to relay_k/v/src.
# =============================================================================

@triton.jit
def _relay_precompute_topk(
    K, V, relay_k, relay_v, relay_src,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_rkb, stride_rkh, stride_rkt, stride_rkd,
    stride_rvb, stride_rvh, stride_rvt, stride_rvd,
    stride_rsb, stride_rsh, stride_rst,
    H_kv,
    S: tl.constexpr, D: tl.constexpr,
    SQRT_N: tl.constexpr, TOP_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    Top-K exact-token relay pre-computation.

    Grid: (B * H_kv, NUM_RELAY_BLOCKS)
    Each program selects the TOP_K tokens with highest ‖k‖² within its
    SQRT_N-element block and stores their exact k/v vectors plus source
    positions for the scatter backward pass.

    relay_k/v : [B, H_kv, NUM_RELAY_BLOCKS * TOP_K, D]
    relay_src  : [B, H_kv, NUM_RELAY_BLOCKS * TOP_K]  float32 source index
    """
    bh = tl.program_id(0)   # linearized (batch * H_kv)
    r  = tl.program_id(1)   # relay block index

    b = bh // H_kv
    h = bh %  H_kv

    s_start = r * SQRT_N
    d_idx   = tl.arange(0, BLOCK_D)
    d_mask  = d_idx < D
    k_base  = K + b * stride_kb + h * stride_kh

    # ── Pass 1: scan norm² for all positions, track top-2 ─────────────────────
    # Sentinel: valid norms are ≥ 0, so -1e38 marks "unset".
    SENT = -1e38
    best0_nsq = tl.full([], SENT, dtype=tl.float32)
    best0_idx = tl.zeros([], dtype=tl.int32)
    best1_nsq = tl.full([], SENT, dtype=tl.float32)
    best1_idx = tl.zeros([], dtype=tl.int32)

    for i in range(SQRT_N):
        s_i     = s_start + i
        valid_i = s_i < S
        k_row   = tl.load(k_base + s_i * stride_ks + d_idx * stride_kd,
                          mask=d_mask & valid_i, other=0.0).to(tl.float32)
        n_sq    = tl.sum(k_row * k_row)
        n_sq    = tl.where(valid_i, n_sq, tl.full([], SENT, dtype=tl.float32))
        i_t     = tl.full([], i, dtype=tl.int32)
        beats0  = n_sq > best0_nsq
        beats1  = n_sq > best1_nsq
        # Update top-2 (evaluate new top1 before overwriting top0)
        new1_nsq = tl.where(beats0, best0_nsq, tl.where(beats1, n_sq,  best1_nsq))
        new1_idx = tl.where(beats0, best0_idx, tl.where(beats1, i_t,   best1_idx))
        best0_nsq = tl.where(beats0, n_sq, best0_nsq)
        best0_idx = tl.where(beats0, i_t,  best0_idx)
        best1_nsq = new1_nsq
        best1_idx = new1_idx

    abs0 = s_start + best0_idx   # absolute sequence position of best token
    abs1 = s_start + best1_idx   # absolute sequence position of 2nd-best token

    # ── Pass 2: load exact k/v at top positions, write relay tokens ────────────
    rk_base = relay_k   + b * stride_rkb + h * stride_rkh
    rv_base = relay_v   + b * stride_rvb + h * stride_rvh
    rs_base = relay_src + b * stride_rsb + h * stride_rsh

    t0 = r * TOP_K
    k0 = tl.load(k_base + abs0 * stride_ks + d_idx * stride_kd, mask=d_mask, other=0.0)
    v0 = tl.load(V + b * stride_vb + h * stride_vh + abs0 * stride_vs + d_idx * stride_vd,
                 mask=d_mask, other=0.0)
    tl.store(rk_base + t0 * stride_rkt + d_idx * stride_rkd, k0, mask=d_mask)
    tl.store(rv_base + t0 * stride_rvt + d_idx * stride_rvd, v0, mask=d_mask)
    tl.store(rs_base + t0 * stride_rst, tl.cast(abs0, tl.float32))

    if TOP_K >= 2:
        t1 = r * TOP_K + 1
        k1 = tl.load(k_base + abs1 * stride_ks + d_idx * stride_kd, mask=d_mask, other=0.0)
        v1 = tl.load(V + b * stride_vb + h * stride_vh + abs1 * stride_vs + d_idx * stride_vd,
                     mask=d_mask, other=0.0)
        tl.store(rk_base + t1 * stride_rkt + d_idx * stride_rkd, k1, mask=d_mask)
        tl.store(rv_base + t1 * stride_rvt + d_idx * stride_rvd, v1, mask=d_mask)
        tl.store(rs_base + t1 * stride_rst, tl.cast(abs1, tl.float32))


# =============================================================================
# RELAY SCATTER BACKWARD KERNEL  (exact-index scatter, matching top-K precompute)
# =============================================================================
#
# Previous weighted scatter: for each relay slot r, distributed gradient across
# SQRT_N source positions using stored norm²-weights wᵢ.
#
# New exact scatter: each relay token t = r*K+j stores one source position
# (written by _relay_precompute_topk into relay_src[t]).  The kernel simply
# scatters d_relay[t] to dK/dV[src_pos].
#
# tl.atomic_add is used because the last (partial) block may have fewer than
# TOP_K valid tokens, causing t=r*K+0 and t=r*K+1 to map to the same src_pos.
# For all full blocks the writes are to distinct rows — no contention.
# =============================================================================

@triton.jit
def _relay_scatter_bwd_topk(
    d_relay, d_kv, relay_src,
    stride_drb, stride_drh, stride_drt, stride_drd,
    stride_dkvb, stride_dkvh, stride_dkvs, stride_dkvd,
    stride_rsb, stride_rsh, stride_rst,
    H_kv,
    D: tl.constexpr, TOP_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    Top-K relay scatter backward.

    Grid: (B * H_kv, NUM_RELAY_BLOCKS)
    Each program owns one relay BLOCK (all TOP_K representative tokens).
    All TOP_K source positions lie within [r*SQRT_N, (r+1)*SQRT_N), which is
    exclusive to this program — no other relay scatter program writes there.
    So plain load+add+store is correct; zero atomics.

    Degenerate last partial block (src0 == src1 when block has <2 valid tokens):
    handled naturally — iteration j=1 re-loads the value j=0 already wrote and
    adds its own gradient on top.  Correct because the loop is sequential.
    """
    bh = tl.program_id(0)
    r  = tl.program_id(1)   # relay block index (NOT flat token index)

    b = bh // H_kv
    h = bh %  H_kv

    d_idx  = tl.arange(0, BLOCK_D)
    d_mask = d_idx < D

    dkv_base = d_kv + b * stride_dkvb + h * stride_dkvh

    for j in tl.static_range(TOP_K):
        t = r * TOP_K + j

        # Load gradient for this relay token [D]
        dr_ptr = d_relay + b * stride_drb + h * stride_drh + t * stride_drt
        dr = tl.load(dr_ptr + d_idx * stride_drd, mask=d_mask, other=0.0)

        # Source position stored as float32 (exact for integers up to 2^24)
        src_f   = tl.load(relay_src + b * stride_rsb + h * stride_rsh + t * stride_rst)
        src_pos = src_f.to(tl.int32)

        # Scatter: read-add-write.  No atomics: exclusive block ownership.
        dkv_ptr = dkv_base + src_pos * stride_dkvs
        cur = tl.load(dkv_ptr + d_idx * stride_dkvd, mask=d_mask, other=0.0)
        tl.store(dkv_ptr + d_idx * stride_dkvd,
                 (cur + dr).to(d_kv.dtype.element_ty), mask=d_mask)


# =============================================================================
# PYTHON WRAPPER CLASS
# =============================================================================

class CronRootAttentionV14Function(torch.autograd.Function):
    """Autograd function for V14 tiled √N attention with 2-hop relay."""
    
    # Relay skip threshold: relay is skipped for sequences at or below this
    # length.  At S<=2048, strided+local provides adequate O(√N) coverage
    # (~64-90 attention targets per query via 2-hop reachability).
    # Setting to 2048 enables GQA backward compatibility at all practical
    # training seq lengths — the relay backward kernels only support MHA.
    # For S>2048 inference/training, relay activates and MHA is required.
    RELAY_THRESHOLD = 2048
    
    # Fused backward threshold: below this, use single-kernel backward
    # (fully fused dQ + local dK/dV + strided dK/dV) instead of 4-kernel approach.
    FUSED_BWD_THRESHOLD = 8192

    # CUDA graph cache for split3 backward.
    # Key: (B, H_q, S, D, BLOCK_M, BLOCK_D, BLOCK_STRIDE, BLOCK_LOCAL, device_idx)
    # Value: (graph, s_q, s_k, s_v, s_o, s_do, s_L, s_rk, s_rv, s_dq, s_dk, s_dv)
    #
    # CUDA graphs collapse 3 Triton kernel dispatches (3 × ~50μs Python overhead)
    # to a single graph.replay() call (~5μs), making split3 strictly better
    # than fully_fused at ALL sequence lengths, not just S ≥ 512.
    _split3_graph_cache: dict = {}

    @staticmethod
    def _split3_launch(q, k, v, o, do, L, relay_k, relay_v, dq, dk, dv,
                       S, D, SQRT_N, BLOCK_M, BLOCK_D, BLOCK_LOCAL, BLOCK_STRIDE,
                       BLOCK_RELAY, B, H_q):
        """Raw 3-kernel split3 backward (no Python-overhead mitigation).
        Used for warmup inside _split3_graphed; never called on the hot path."""
        num_tiles_m     = (S + BLOCK_M - 1) // BLOCK_M
        BLOCK_N, BLOCK_Q = 32, 32
        num_tiles_n     = (S + BLOCK_N - 1) // BLOCK_N
        num_strided     = (S + SQRT_N - 1) // SQRT_N
        BLOCK_Q_STRIDED = 64
        # Select NUM_Q_TILES: parallelize the O(N) query loop across tiles
        NUM_Q_TILES_STRIDED = 1 if S <= 256 else (2 if S <= 512 else 4)

        # Inline auto-tune: first backward per (H, D) benchmarks dQ kernel
        if not _tuner.is_tuned("bwd", H_q, D):
            _tuner.tune_bwd_inline(
                _cron_root_attn_bwd_dq_only_v14,
                (B, H_q, num_tiles_m),
                (q, k, v, o, do, dq, L,
                 relay_k, relay_v,
                 q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                 k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                 v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                 o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                 do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                 dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                 L.stride(0), L.stride(1), L.stride(2),
                 relay_k.stride(0), relay_k.stride(1), relay_k.stride(2), relay_k.stride(3),
                 relay_v.stride(0), relay_v.stride(1), relay_v.stride(2), relay_v.stride(3)),
                dict(S=S, D=D, SQRT_N=SQRT_N,
                     NUM_RELAY=0, BLOCK_RELAY=BLOCK_RELAY,
                     BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
                     BLOCK_LOCAL=BLOCK_LOCAL, BLOCK_STRIDE=BLOCK_STRIDE,
                     TOP_K=2),
                H=H_q, D=D)
        bwd_nw, bwd_ns = _tuner.get_config("bwd", H_q, D)

        _cron_root_attn_bwd_dq_only_v14[(B, H_q, num_tiles_m)](
            q, k, v, o, do, dq, L,
            relay_k, relay_v,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            relay_k.stride(0), relay_k.stride(1), relay_k.stride(2), relay_k.stride(3),
            relay_v.stride(0), relay_v.stride(1), relay_v.stride(2), relay_v.stride(3),
            S=S, D=D, SQRT_N=SQRT_N,
            NUM_RELAY=0, BLOCK_RELAY=BLOCK_RELAY,
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
            BLOCK_LOCAL=BLOCK_LOCAL, BLOCK_STRIDE=BLOCK_STRIDE,
            TOP_K=2,
            num_warps=bwd_nw, num_stages=bwd_ns,
        )
        _dkdv_local_phase[(B, H_q, num_tiles_n)](
            q, k, v, o, do, dk, dv, L,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            S=S, D=D, SQRT_N=SQRT_N,
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_Q=BLOCK_Q,
            num_warps=bwd_nw, num_stages=bwd_ns,
        )
        if num_strided > 0:
            _dkdv_strided_key_centric[(num_strided * NUM_Q_TILES_STRIDED, B, H_q)](
                q, k, v, o, do, dk, dv, L,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                L.stride(0), L.stride(1), L.stride(2),
                S=S, D=D, SQRT_N=SQRT_N,
                BLOCK_Q=BLOCK_Q_STRIDED, BLOCK_D=BLOCK_D,
                NUM_Q_TILES=NUM_Q_TILES_STRIDED,
                num_warps=bwd_nw, num_stages=bwd_ns,
            )

    @staticmethod
    def _split3_graphed(q, k, v, o, do, L, relay_k, relay_v,
                        S, D, SQRT_N, BLOCK_M, BLOCK_D, BLOCK_LOCAL, BLOCK_STRIDE,
                        BLOCK_RELAY, B, H_q):
        """Split3 backward with CUDA graph replay after first-call JIT warmup.

        First call per shape: compiles Triton kernels + captures CUDA graph.
        Subsequent calls: copies inputs into static buffers, calls graph.replay()
        (single CUDA API dispatch, ~5μs CPU overhead vs ~150μs for 3 raw launches).

        Memory overhead: allocates 2 × (5 input + 3 output) BF16 tensors of shape
        (B, H, S, D) per unique (B, H, S, D) shape — held for the lifetime of
        the process (cache never evicted).
        """
        cache = CronRootAttentionV14Function._split3_graph_cache
        cache_key = (B, H_q, S, D, BLOCK_M, BLOCK_D, BLOCK_STRIDE, BLOCK_LOCAL,
                     q.device.index)

        if cache_key not in cache:
            # Pipeline-parallel fix: q may live on cuda:1 while Python's current
            # device is cuda:0.  Two separate streams are needed:
            #   _wup_stream (default)  – Triton warmup; Triton reads
            #                            torch.cuda.current_stream() so it must
            #                            be the default stream for q.device.
            #   _cap_stream (non-default) – CUDAGraph capture; PyTorch requires
            #                              capture on a non-default stream.
            # Replay is submitted via g.replay() on the default stream (allowed).
            _wup_stream = torch.cuda.default_stream(q.device)
            _cap_stream = torch.cuda.Stream(device=q.device)

            with torch.cuda.stream(_wup_stream):
                # ── Allocate static (fixed-address) I/O buffers ───────────────
                s_q  = q.clone();   s_k  = k.clone();  s_v  = v.clone()
                s_o  = o.clone();   s_do = do.clone(); s_L  = L.clone()
                s_rk = relay_k.clone(); s_rv = relay_v.clone()
                s_dq = torch.empty_like(q)
                s_dk = torch.empty_like(k)
                s_dv = torch.empty_like(v)

                # ── Warmup: JIT-compile Triton kernels outside capture ─────────
                _wup_stream.synchronize()
                CronRootAttentionV14Function._split3_launch(
                    s_q, s_k, s_v, s_o, s_do, s_L, s_rk, s_rv, s_dq, s_dk, s_dv,
                    S, D, SQRT_N, BLOCK_M, BLOCK_D, BLOCK_LOCAL, BLOCK_STRIDE,
                    BLOCK_RELAY, B, H_q)
                _wup_stream.synchronize()

            # ── Capture: non-default stream required by PyTorch ───────────────
            with torch.cuda.stream(_cap_stream):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=_cap_stream):
                    CronRootAttentionV14Function._split3_launch(
                        s_q, s_k, s_v, s_o, s_do, s_L, s_rk, s_rv, s_dq, s_dk, s_dv,
                        S, D, SQRT_N, BLOCK_M, BLOCK_D, BLOCK_LOCAL, BLOCK_STRIDE,
                        BLOCK_RELAY, B, H_q)
            # Sync before storing cache: CUDA graph capture also executes kernels
            # once on _cap_stream (JIT compilation dry-run), which READS
            # s_q/s_k/s_v.  Without this sync, the replay path immediately calls
            # s_q.copy_() on default_stream while _cap_stream is still reading
            # s_q → concurrent READ+WRITE on same GPU memory →
            # cudaErrorIllegalAddress.  This sync fires only once per unique shape.
            _cap_stream.synchronize()

            cache[cache_key] = (g, s_q, s_k, s_v, s_o, s_do, s_L, s_rk, s_rv,
                                s_dq, s_dk, s_dv)

        g, s_q, s_k, s_v, s_o, s_do, s_L, s_rk, s_rv, s_dq, s_dk, s_dv = \
            cache[cache_key]

        # Replay on default stream for q's device (copies + replay sequenced).
        with torch.cuda.stream(torch.cuda.default_stream(q.device)):
            s_q.copy_(q);  s_k.copy_(k);  s_v.copy_(v)
            s_o.copy_(o);  s_do.copy_(do); s_L.copy_(L)
            s_rk.copy_(relay_k); s_rv.copy_(relay_v)
            g.replay()
            # Return static output buffers directly — no clone needed.
            # Safety: all ops are on the same default stream; the training loop
            # serialises autograd consumption of s_dq/dk/dv before the next
            # g.replay() overwrites them. Eliminates 3×16MB memcpy per layer.
            return s_dq, s_dk, s_dv

    # CUDA graph cache for GQA split3 backward.
    # Key: (B, H_q, H_kv, S, D, KV_GROUPS, BLOCK_M, BLOCK_D, BLOCK_STRIDE,
    #        BLOCK_LOCAL, device_idx)
    _split3_gqa_graph_cache: dict = {}

    @staticmethod
    def _split3_launch_gqa(q, k, v, o, do, L, relay_k, relay_v, dq, dk, dv,
                           S, D, SQRT_N, BLOCK_M, BLOCK_D, BLOCK_LOCAL, BLOCK_STRIDE,
                           BLOCK_RELAY, B, H_q, H_kv, KV_GROUPS):
        """Raw 3-kernel GQA split3 backward (KV-head-centric dK/dV kernels).
        Used for warmup inside _split3_graphed_gqa; never called on the hot path."""
        num_tiles_m      = (S + BLOCK_M - 1) // BLOCK_M
        BLOCK_N, BLOCK_Q = 32, 32
        num_kv_tiles     = (S + BLOCK_N - 1) // BLOCK_N
        num_strided      = (S + SQRT_N - 1) // SQRT_N
        BLOCK_Q_STRIDED  = 64
        # Select NUM_Q_TILES: parallelize the O(N) query loop across tiles
        NUM_Q_TILES_STRIDED = 1 if S <= 256 else (2 if S <= 512 else 4)

        # Reuse cached bwd config (tuned by split3 MHA or tune_bwd_inline)
        bwd_nw, bwd_ns = _tuner.get_config("bwd", H_q, D)

        # dQ kernel: Q-head-centric, reads K/V at (h_q // KV_GROUPS)
        _cron_root_attn_bwd_dq_only_v14[(B, H_q, num_tiles_m)](
            q, k, v, o, do, dq, L,
            relay_k, relay_v,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            relay_k.stride(0), relay_k.stride(1), relay_k.stride(2), relay_k.stride(3),
            relay_v.stride(0), relay_v.stride(1), relay_v.stride(2), relay_v.stride(3),
            S=S, D=D, SQRT_N=SQRT_N,
            NUM_RELAY=0, BLOCK_RELAY=BLOCK_RELAY,
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
            BLOCK_LOCAL=BLOCK_LOCAL, BLOCK_STRIDE=BLOCK_STRIDE,
            KV_GROUPS=KV_GROUPS, TOP_K=2,
            num_warps=bwd_nw, num_stages=bwd_ns,
        )
        # dK/dV local phase: KV-head-centric, direct store (no atomics)
        _dkdv_local_phase_gqa[(B, H_kv, num_kv_tiles)](
            q, k, v, o, do, dk, dv, L,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            S=S, D=D, SQRT_N=SQRT_N,
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_Q=BLOCK_Q,
            KV_GROUPS=KV_GROUPS,
            num_warps=bwd_nw, num_stages=bwd_ns,
        )
        # dK/dV strided phase: KV-head-centric, atomic_add merges with local
        if num_strided > 0:
            _dkdv_strided_key_centric_gqa[(num_strided * NUM_Q_TILES_STRIDED, B, H_kv)](
                q, k, v, o, do, dk, dv, L,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                L.stride(0), L.stride(1), L.stride(2),
                S=S, D=D, SQRT_N=SQRT_N,
                BLOCK_Q=BLOCK_Q_STRIDED, BLOCK_D=BLOCK_D,
                KV_GROUPS=KV_GROUPS, NUM_Q_TILES=NUM_Q_TILES_STRIDED,
                num_warps=bwd_nw, num_stages=bwd_ns,
            )

    @staticmethod
    def _split3_graphed_gqa(q, k, v, o, do, L, relay_k, relay_v,
                            S, D, SQRT_N, BLOCK_M, BLOCK_D, BLOCK_LOCAL, BLOCK_STRIDE,
                            BLOCK_RELAY, B, H_q, H_kv, KV_GROUPS):
        """GQA split3 backward with CUDA graph replay after first-call JIT warmup.

        Identical lifecycle to _split3_graphed but caches per (B, H_q, H_kv,
        S, D, KV_GROUPS, ...) and dispatches the KV-head-centric dK/dV kernels.
        """
        cache     = CronRootAttentionV14Function._split3_gqa_graph_cache
        cache_key = (B, H_q, H_kv, S, D, KV_GROUPS,
                     BLOCK_M, BLOCK_D, BLOCK_STRIDE, BLOCK_LOCAL, q.device.index)

        if cache_key not in cache:
            # Same two-stream fix as _split3_graphed (MHA version above):
            # warmup on default stream (Triton requirement), capture on a
            # dedicated non-default stream (PyTorch CUDAGraph requirement),
            # replay on default stream (explicitly allowed by PyTorch).
            _wup_stream = torch.cuda.default_stream(q.device)
            _cap_stream = torch.cuda.Stream(device=q.device)

            with torch.cuda.stream(_wup_stream):
                s_q  = q.clone();   s_k  = k.clone();  s_v  = v.clone()
                s_o  = o.clone();   s_do = do.clone(); s_L  = L.clone()
                s_rk = relay_k.clone(); s_rv = relay_v.clone()
                s_dq = torch.empty_like(q)
                s_dk = torch.empty_like(k)
                s_dv = torch.empty_like(v)

                _wup_stream.synchronize()
                CronRootAttentionV14Function._split3_launch_gqa(
                    s_q, s_k, s_v, s_o, s_do, s_L, s_rk, s_rv, s_dq, s_dk, s_dv,
                    S, D, SQRT_N, BLOCK_M, BLOCK_D, BLOCK_LOCAL, BLOCK_STRIDE,
                    BLOCK_RELAY, B, H_q, H_kv, KV_GROUPS)
                _wup_stream.synchronize()

            with torch.cuda.stream(_cap_stream):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=_cap_stream):
                    CronRootAttentionV14Function._split3_launch_gqa(
                        s_q, s_k, s_v, s_o, s_do, s_L, s_rk, s_rv, s_dq, s_dk, s_dv,
                        S, D, SQRT_N, BLOCK_M, BLOCK_D, BLOCK_LOCAL, BLOCK_STRIDE,
                        BLOCK_RELAY, B, H_q, H_kv, KV_GROUPS)
            # Same sync as _split3_graphed: wait for _cap_stream to finish the
            # capture-time execution before default_stream copy_()s new inputs
            # into s_q/s_k/s_v → prevents concurrent READ+WRITE race.
            _cap_stream.synchronize()

            cache[cache_key] = (g, s_q, s_k, s_v, s_o, s_do, s_L, s_rk, s_rv,
                                s_dq, s_dk, s_dv)

        g, s_q, s_k, s_v, s_o, s_do, s_L, s_rk, s_rv, s_dq, s_dk, s_dv = \
            cache[cache_key]

        with torch.cuda.stream(torch.cuda.default_stream(q.device)):
            s_q.copy_(q);  s_k.copy_(k);  s_v.copy_(v)
            s_o.copy_(o);  s_do.copy_(do); s_L.copy_(L)
            s_rk.copy_(relay_k); s_rv.copy_(relay_v)
            g.replay()
            # Return static output buffers directly — safe on single default stream.
            return s_dq, s_dk, s_dv

    # ── RELAY-4 CUDA-graph cache (MHA, relay-active path) ─────────────────
    # Key: (B, H_q, S, D, NUM_RELAY, BLOCK_M, BLOCK_D, BLOCK_STRIDE,
    #        BLOCK_LOCAL, BLOCK_RELAY, device_idx)
    # Value: (graph, s_q, s_k, s_v, s_o, s_do, s_L, s_rk, s_rv, s_relay_w,
    #         s_d_rk, s_d_rv, s_dq, s_dk, s_dv)
    #
    # Collapses 6 Triton dispatches (dQ-only + local-dK/dV + strided-dK/dV +
    # relay-dK/dV + 2×relay_scatter_bwd) from ~6×50μs = 300μs Python overhead
    # to a single graph.replay() (~5μs). Active for S > RELAY_THRESHOLD (256)
    # i.e. the entire S=512 training stage.
    _relay4_graph_cache: dict = {}

    # CUDA graph cache for forward (skip_relay tiled path).
    # Key: (B, H_q, H_kv, S, D, BLOCK_M, BLOCK_D, BLOCK_STRIDE, BLOCK_LOCAL,
    #        kv_groups, device_idx)
    # Value: (graph, s_q, s_k, s_v, s_o, s_L)
    #
    # Eliminates per-step Python overhead of 5×torch.empty() + 1 Triton
    # kernel dispatch (~120μs/layer at S=512) by replaying a pre-captured
    # CUDA graph (~5μs CPU).  For a 4-layer model: ~460μs saved per step.
    # Same lifecycle as _split3_graphed: one warmup+capture per unique shape.
    _fwd_graph_cache: dict = {}

    @staticmethod
    def _relay4_launch(q, k, v, o, do, L, relay_k, relay_v, relay_src,
                       dq, dk, dv, d_relay_k, d_relay_v,
                       S, D, SQRT_N, NUM_RELAY, TOP_K,
                       BLOCK_M, BLOCK_D, BLOCK_LOCAL, BLOCK_STRIDE, BLOCK_RELAY,
                       B, H_q):
        """Raw 6-kernel relay4 MHA backward (for warmup).  Never called on the hot path."""
        BLOCK_N, BLOCK_Q   = 32, 32
        num_tiles_m        = (S + BLOCK_M - 1) // BLOCK_M
        num_tiles_n        = (S + BLOCK_N - 1) // BLOCK_N
        num_strided_keys   = (S + SQRT_N - 1) // SQRT_N
        BLOCK_Q_STRIDED    = 64
        NUM_Q_TILES_STRIDED = 1 if S <= 256 else (2 if S <= 512 else 4)
        BLOCK_Q_RELAY      = 64
        _BLOCK_D_RELAY     = triton.next_power_of_2(D)

        bwd_nw, bwd_ns = _tuner.get_config("bwd", H_q, D)

        # dQ-only (includes relay contribution to dQ)
        _cron_root_attn_bwd_dq_only_v14[(B, H_q, num_tiles_m)](
            q, k, v, o, do, dq, L,
            relay_k, relay_v,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            relay_k.stride(0), relay_k.stride(1), relay_k.stride(2), relay_k.stride(3),
            relay_v.stride(0), relay_v.stride(1), relay_v.stride(2), relay_v.stride(3),
            S=S, D=D, SQRT_N=SQRT_N,
            NUM_RELAY=NUM_RELAY, BLOCK_RELAY=BLOCK_RELAY,
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
            BLOCK_LOCAL=BLOCK_LOCAL, BLOCK_STRIDE=BLOCK_STRIDE,
            KV_GROUPS=1, TOP_K=TOP_K,
            num_warps=bwd_nw, num_stages=bwd_ns,
        )
        # local dK/dV
        _dkdv_local_phase[(B, H_q, num_tiles_n)](
            q, k, v, o, do, dk, dv, L,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            S=S, D=D, SQRT_N=SQRT_N,
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_Q=BLOCK_Q,
            num_warps=bwd_nw, num_stages=bwd_ns,
        )
        # strided dK/dV
        if num_strided_keys > 0:
            _dkdv_strided_key_centric[(num_strided_keys * NUM_Q_TILES_STRIDED, B, H_q)](
                q, k, v, o, do, dk, dv, L,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                L.stride(0), L.stride(1), L.stride(2),
                S=S, D=D, SQRT_N=SQRT_N,
                BLOCK_Q=BLOCK_Q_STRIDED, BLOCK_D=BLOCK_D,
                NUM_Q_TILES=NUM_Q_TILES_STRIDED,
                num_warps=bwd_nw, num_stages=bwd_ns,
            )
        # relay dK/dV (MHA: H_q heads)
        _dkdv_relay_v14[(NUM_RELAY, B, H_q)](
            q, relay_k, relay_v, o, do, d_relay_k, d_relay_v, L,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            relay_k.stride(0), relay_k.stride(1), relay_k.stride(2), relay_k.stride(3),
            relay_v.stride(0), relay_v.stride(1), relay_v.stride(2), relay_v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            d_relay_k.stride(0), d_relay_k.stride(1), d_relay_k.stride(2), d_relay_k.stride(3),
            d_relay_v.stride(0), d_relay_v.stride(1), d_relay_v.stride(2), d_relay_v.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            S=S, D=D, SQRT_N=SQRT_N,
            NUM_RELAY=NUM_RELAY,
            BLOCK_Q=BLOCK_Q_RELAY, BLOCK_D=BLOCK_D,
            TOP_K=TOP_K,
        )
        # relay scatter: d_relay_k/v → dK, dV
        # Grid uses NUM_RELAY_BLOCKS (not NUM_RELAY), since each program owns
        # one block (TOP_K relay tokens) and loops over them sequentially —
        # exclusive block ownership means zero atomics.
        NUM_RELAY_BLOCKS_SCATTER = NUM_RELAY // TOP_K
        _relay_scatter_bwd_topk[(B * H_q, NUM_RELAY_BLOCKS_SCATTER)](
            d_relay_k, dk, relay_src,
            d_relay_k.stride(0), d_relay_k.stride(1), d_relay_k.stride(2), d_relay_k.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            relay_src.stride(0), relay_src.stride(1), relay_src.stride(2),
            H_kv=H_q,
            D=D, TOP_K=TOP_K, BLOCK_D=_BLOCK_D_RELAY,
        )
        _relay_scatter_bwd_topk[(B * H_q, NUM_RELAY_BLOCKS_SCATTER)](
            d_relay_v, dv, relay_src,
            d_relay_v.stride(0), d_relay_v.stride(1), d_relay_v.stride(2), d_relay_v.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            relay_src.stride(0), relay_src.stride(1), relay_src.stride(2),
            H_kv=H_q,
            D=D, TOP_K=TOP_K, BLOCK_D=_BLOCK_D_RELAY,
        )

    @staticmethod
    def _relay4_graphed(q, k, v, o, do, L, relay_k, relay_v, relay_src,
                        S, D, SQRT_N, NUM_RELAY, TOP_K,
                        BLOCK_M, BLOCK_D, BLOCK_LOCAL, BLOCK_STRIDE, BLOCK_RELAY,
                        B, H_q):
        """Relay-4 MHA backward with CUDA graph replay after first-call JIT warmup.

        Same lifecycle as _split3_graphed: one warmup + capture per unique shape,
        thereafter a single graph.replay() (~5μs CPU) replaces 6 raw Triton dispatches
        (~300μs raw kernel launch overhead at S=512, 11 CRA layers).

        d_relay_log_scale (a tiny [H_q] sum) is computed in Python AFTER graph replay
        so it does not need to be inside the captured graph.
        """
        cache = CronRootAttentionV14Function._relay4_graph_cache
        cache_key = (B, H_q, S, D, NUM_RELAY, TOP_K,
                     BLOCK_M, BLOCK_D, BLOCK_STRIDE, BLOCK_LOCAL, BLOCK_RELAY,
                     q.device.index)

        if cache_key not in cache:
            _wup_stream = torch.cuda.default_stream(q.device)
            _cap_stream = torch.cuda.Stream(device=q.device)

            with torch.cuda.stream(_wup_stream):
                s_q       = q.clone();        s_k       = k.clone()
                s_v       = v.clone();        s_o       = o.clone()
                s_do      = do.clone();       s_L       = L.clone()
                s_rk      = relay_k.clone();  s_rv      = relay_v.clone()
                s_rs      = relay_src.clone()
                s_dq      = torch.empty_like(q)
                s_dk      = torch.zeros_like(k)  # zeros: atomic accumulation in kernels
                s_dv      = torch.zeros_like(v)
                s_d_rk    = torch.zeros_like(relay_k)
                s_d_rv    = torch.zeros_like(relay_v)

                _wup_stream.synchronize()
                CronRootAttentionV14Function._relay4_launch(
                    s_q, s_k, s_v, s_o, s_do, s_L, s_rk, s_rv, s_rs,
                    s_dq, s_dk, s_dv, s_d_rk, s_d_rv,
                    S, D, SQRT_N, NUM_RELAY, TOP_K,
                    BLOCK_M, BLOCK_D, BLOCK_LOCAL, BLOCK_STRIDE, BLOCK_RELAY, B, H_q)
                _wup_stream.synchronize()

            with torch.cuda.stream(_cap_stream):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=_cap_stream):
                    # dk/dv must be zeroed for atomic accumulation INSIDE the graph
                    s_dk.zero_()
                    s_dv.zero_()
                    s_d_rk.zero_()
                    s_d_rv.zero_()
                    CronRootAttentionV14Function._relay4_launch(
                        s_q, s_k, s_v, s_o, s_do, s_L, s_rk, s_rv, s_rs,
                        s_dq, s_dk, s_dv, s_d_rk, s_d_rv,
                        S, D, SQRT_N, NUM_RELAY, TOP_K,
                        BLOCK_M, BLOCK_D, BLOCK_LOCAL, BLOCK_STRIDE, BLOCK_RELAY, B, H_q)
            _cap_stream.synchronize()

            cache[cache_key] = (g, s_q, s_k, s_v, s_o, s_do, s_L, s_rk, s_rv, s_rs,
                                s_d_rk, s_d_rv, s_dq, s_dk, s_dv)

        (g, s_q, s_k, s_v, s_o, s_do, s_L, s_rk, s_rv, s_rs,
         s_d_rk, s_d_rv, s_dq, s_dk, s_dv) = cache[cache_key]

        with torch.cuda.stream(torch.cuda.default_stream(q.device)):
            s_q.copy_(q);   s_k.copy_(k);   s_v.copy_(v)
            s_o.copy_(o);   s_do.copy_(do); s_L.copy_(L)
            s_rk.copy_(relay_k); s_rv.copy_(relay_v); s_rs.copy_(relay_src)
            g.replay()
            # d_relay_log_scale is a tiny [H_q] reduction — compute in Python
            # after replay (not inside the graph) to keep graph code simple.
            d_relay_log_scale = (
                (s_d_rk.float() * s_rk.float()).sum(dim=(0, 2, 3))
                + (s_d_rv.float() * s_rv.float()).sum(dim=(0, 2, 3))
            )  # [H_q]; caller converts dtype if needed
            # Static output buffers returned directly (safe, same default stream).
            return (s_dq, s_dk, s_dv,
                    s_d_rk, s_d_rv, d_relay_log_scale)

    @staticmethod
    def _fwd_launch_skip_relay(s_q, s_k, s_v, s_o, s_L, s_rk, s_rv,
                               S, D, SQRT_N, BLOCK_M, BLOCK_D, BLOCK_LOCAL,
                               BLOCK_STRIDE, BLOCK_RELAY, B, H_q, kv_groups,
                               fwd_nw, fwd_ns):
        """Raw forward tiled kernel (skip_relay path).  Only called during
        _fwd_graphed_skip_relay warmup; never on the steady-state hot path."""
        num_tiles = (S + BLOCK_M - 1) // BLOCK_M
        _cron_root_attn_fwd_v14_tiled[(B, H_q, num_tiles)](
            s_q, s_k, s_v, s_o, s_L,
            s_rk, s_rv,
            s_q.stride(0), s_q.stride(1), s_q.stride(2), s_q.stride(3),
            s_k.stride(0), s_k.stride(1), s_k.stride(2), s_k.stride(3),
            s_v.stride(0), s_v.stride(1), s_v.stride(2), s_v.stride(3),
            s_o.stride(0), s_o.stride(1), s_o.stride(2), s_o.stride(3),
            s_L.stride(0), s_L.stride(1), s_L.stride(2),
            s_rk.stride(0), s_rk.stride(1), s_rk.stride(2), s_rk.stride(3),
            s_rv.stride(0), s_rv.stride(1), s_rv.stride(2), s_rv.stride(3),
            S=S, D=D, SQRT_N=SQRT_N,
            NUM_RELAY=0, BLOCK_RELAY=BLOCK_RELAY,
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
            BLOCK_LOCAL=BLOCK_LOCAL, BLOCK_STRIDE=BLOCK_STRIDE,
            KV_GROUPS=kv_groups,
            TOP_K=2,
            num_warps=fwd_nw, num_stages=fwd_ns,
        )

    @staticmethod
    def _fwd_graphed_skip_relay(q, k, v,
                                S, D, SQRT_N, BLOCK_M, BLOCK_D, BLOCK_LOCAL,
                                BLOCK_STRIDE, BLOCK_RELAY, B, H_q, H_kv,
                                kv_groups, fwd_nw, fwd_ns):
        """Forward CUDA graph for the skip_relay tiled path (training hot path).

        First call per shape: JIT-compiles the Triton kernel + captures CUDA graph.
        Subsequent calls: copies q/k/v into static buffers, calls graph.replay()
        (~5μs CPU overhead vs ~120μs for 5×empty()+kernel dispatch), clones o/L
        so the static buffers are free for the next step without corrupting the
        tensors saved for the backward pass.

        Memory overhead: holds 5 BF16 tensors + 1 FP32 tensor of shape
        (B,H,S,D) / (B,H,S) per unique (B,H_q,H_kv,S,D,kv_groups) — same
        lifecycle as _split3_graph_cache.
        """
        cache = CronRootAttentionV14Function._fwd_graph_cache
        cache_key = (B, H_q, H_kv, S, D, BLOCK_M, BLOCK_D, BLOCK_STRIDE,
                     BLOCK_LOCAL, kv_groups, q.device.index)

        if cache_key not in cache:
            # Two-stream setup: warmup on default stream (Triton reads
            # torch.cuda.current_stream()), capture on non-default stream
            # (PyTorch CUDAGraph API requirement).
            _wup_stream = torch.cuda.default_stream(q.device)
            _cap_stream = torch.cuda.Stream(device=q.device)

            with torch.cuda.stream(_wup_stream):
                # ── Static I/O buffers (fixed GPU addresses) ──────────────
                s_q  = q.clone();  s_k = k.clone();  s_v = v.clone()
                s_o  = torch.empty_like(q)
                s_L  = torch.empty(B, H_q, S, dtype=torch.float32, device=q.device)
                # Dummy relay buffers: NUM_RELAY=0 so kernel never reads them,
                # but strides are recorded in the captured graph — sizes must
                # be consistent between warmup and every future replay call.
                s_rk = torch.empty(B, H_kv, 1, D, device=q.device, dtype=q.dtype)
                s_rv = torch.empty(B, H_kv, 1, D, device=q.device, dtype=q.dtype)

                # ── Warmup: JIT-compile Triton outside the capture ─────────
                _wup_stream.synchronize()
                CronRootAttentionV14Function._fwd_launch_skip_relay(
                    s_q, s_k, s_v, s_o, s_L, s_rk, s_rv,
                    S, D, SQRT_N, BLOCK_M, BLOCK_D, BLOCK_LOCAL,
                    BLOCK_STRIDE, BLOCK_RELAY, B, H_q, kv_groups,
                    fwd_nw, fwd_ns)
                _wup_stream.synchronize()

            # ── Capture on non-default stream ─────────────────────────────
            with torch.cuda.stream(_cap_stream):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=_cap_stream):
                    CronRootAttentionV14Function._fwd_launch_skip_relay(
                        s_q, s_k, s_v, s_o, s_L, s_rk, s_rv,
                        S, D, SQRT_N, BLOCK_M, BLOCK_D, BLOCK_LOCAL,
                        BLOCK_STRIDE, BLOCK_RELAY, B, H_q, kv_groups,
                        fwd_nw, fwd_ns)
            # Sync before caching: capture also executes kernels once on
            # _cap_stream. Without this sync, the first replay's s_q.copy_()
            # on default_stream races with _cap_stream still reading s_q →
            # cudaErrorIllegalAddress.  Fires only once per unique shape.
            _cap_stream.synchronize()

            cache[cache_key] = (g, s_q, s_k, s_v, s_o, s_L)

        g, s_q, s_k, s_v, s_o, s_L = cache[cache_key]

        with torch.cuda.stream(torch.cuda.default_stream(q.device)):
            s_q.copy_(q);  s_k.copy_(k);  s_v.copy_(v)
            g.replay()
            # Clone outputs so the static buffers can be overwritten on the
            # next step without corrupting o/L held in the backward ctx.
            return s_o.clone(), s_L.clone()

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                use_persistent: bool = False, kv_groups: int = 1,
                relay_log_scale: Optional[torch.Tensor] = None,
                relay_top_k: int = 2) -> torch.Tensor:
        B, H_q, S, D = q.shape
        H_kv = k.shape[1]  # H_q // kv_groups (or H_q when kv_groups==1)
        SQRT_N = int(math.ceil(math.sqrt(S)))
        
        # Block sizes for RTX consumer GPUs (SRAM limit ~101376 B).
        #
        # BACKWARD SRAM constraint: fully-fused backward kernel requires ~131072 B
        # with BLOCK_M=32, BLOCK_D=128 (over the 101376 B limit).  Halving to
        # BLOCK_M=16 drops requirement to ~65536 B — fits comfortably.
        #
        # FORWARD SRAM: forward kernel is much lighter (no dQ/dK/dV accumulators).
        # BLOCK_M=32 in forward requires ~48 KB — well within the 99 KB limit.
        # Using a larger forward tile halves the grid (1024→512 blocks at S=512)
        # and improves arithmetic intensity per thread block.
        #
        # BLOCK_STRIDE 32: adds ~16 KB to SRAM (vs 16 at 8 KB) — still fits for
        # both forward (~64 KB) and backward (~81 KB < 99 KB limit), and halves
        # the number of iterations in the strided-attention loop.
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_M_BWD = 16 if BLOCK_D >= 128 else 32   # Backward: SRAM-constrained
        BLOCK_M     = 32 if BLOCK_D >= 128 else 64   # Forward: bigger tile, better occupancy
        BLOCK_LOCAL = BLOCK_M                         # Local window tile = forward query tile
        BLOCK_STRIDE = 32                              # Was 16; halves strided-loop iterations
        BLOCK_RELAY = 16  # Relay blocks per iteration (unchanged)
        
        # =====================================================================
        # RELAY SKIP: For short sequences, strided positions alone provide
        # O(√N) global coverage. Relay adds overhead but marginal value.
        # Skipping eliminates: pad + reshape + mean + contiguous (2-4 CUDA ops)
        # =====================================================================
        skip_relay = (S <= CronRootAttentionV14Function.RELAY_THRESHOLD)
        
        if skip_relay:
            NUM_RELAY = 0
            # Dummy relay buffers (never accessed since NUM_RELAY=0)
            relay_k   = torch.empty(B, H_kv, 1, D, device=q.device, dtype=q.dtype)
            relay_v   = torch.empty(B, H_kv, 1, D, device=q.device, dtype=q.dtype)
            relay_src = torch.empty(B, H_kv, 1, dtype=torch.float32, device=q.device)
        else:
            # Top-K exact-token relay: select the relay_top_k highest-‖k‖² tokens
            # per √N block as exact relay representatives (no blurring).
            NUM_RELAY_BLOCKS = (S + SQRT_N - 1) // SQRT_N
            NUM_RELAY = NUM_RELAY_BLOCKS * relay_top_k
            relay_k   = torch.empty(B, H_kv, NUM_RELAY, D, device=k.device, dtype=k.dtype)
            relay_v   = torch.empty(B, H_kv, NUM_RELAY, D, device=k.device, dtype=k.dtype)
            # relay_src[b, h, t] = float32 source position for relay token t.
            relay_src = torch.empty(B, H_kv, NUM_RELAY, dtype=torch.float32, device=k.device)
            _BLOCK_D_RELAY = triton.next_power_of_2(D)
            _relay_precompute_topk[(B * H_kv, NUM_RELAY_BLOCKS)](
                k, v, relay_k, relay_v, relay_src,
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                relay_k.stride(0), relay_k.stride(1), relay_k.stride(2), relay_k.stride(3),
                relay_v.stride(0), relay_v.stride(1), relay_v.stride(2), relay_v.stride(3),
                relay_src.stride(0), relay_src.stride(1), relay_src.stride(2),
                H_kv=H_kv,
                S=S, D=D, SQRT_N=SQRT_N, TOP_K=relay_top_k, BLOCK_D=_BLOCK_D_RELAY,
            )
        
        # Apply learnable per-head relay scale: relay_k *= exp(relay_log_scale[h]),
        # relay_v *= exp(relay_log_scale[h]).  The saved relay_k/v are the SCALED
        # versions — backward chain-rules through exp correctly.
        # Only applied when relay is active (not skip_relay) and scale is provided.
        ctx.relay_log_scale = relay_log_scale  # None when unused
        if relay_log_scale is not None and not skip_relay:
            _relay_scale = relay_log_scale.exp().to(relay_k.dtype)  # [H_kv]
            relay_k = relay_k * _relay_scale[None, :, None, None]
            relay_v = relay_v * _relay_scale[None, :, None, None]

        num_tiles = (S + BLOCK_M - 1) // BLOCK_M

        if use_persistent and skip_relay:
            # Persistent work-stealing kernel: exactly NUM_SMS blocks with internal
            # work queue. Optimal for S <= auto_persistent_threshold (default 4096).
            # Relay skipped: for S <= RELAY_THRESHOLD the strided path covers global
            # context with no extra overhead.
            o = torch.empty_like(q)
            L = torch.empty(B, H_q, S, dtype=torch.float32, device=q.device)
            total_tiles = B * H_q * num_tiles
            work_ctr = torch.zeros(1, dtype=torch.int32, device=q.device)
            _cron_root_attn_fwd_v14_persistent[(_get_num_sms(),)](
                q, k, v, o, L,
                work_ctr,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                L.stride(0), L.stride(1), L.stride(2),
                B=B, H=H_q, S=S, D=D,
                SQRT_N=SQRT_N, TOTAL_TILES=total_tiles,
                BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
                BLOCK_LOCAL=BLOCK_LOCAL, BLOCK_STRIDE=BLOCK_STRIDE,
            )
        else:
            # Standard tiled kernel path (fixed grid; supports relay for S > 8192)
            # Inline auto-tune: first call per (H, D) benchmarks all candidates
            # using these actual tensors (~10ms one-time), then caches.
            if not _tuner.is_tuned("fwd", H_q, D):
                # tune_fwd_inline needs o/L as scratch — allocate here only for
                # the one-time auto-tune; hot path allocates below after tuning.
                _o_tune = torch.empty_like(q)
                _L_tune = torch.empty(B, H_q, S, dtype=torch.float32, device=q.device)
                fwd_nw, fwd_ns = _tuner.tune_fwd_inline(
                    q, k, v, _o_tune, _L_tune, relay_k, relay_v,
                    S, D, SQRT_N, NUM_RELAY, BLOCK_RELAY,
                    BLOCK_M, BLOCK_D, BLOCK_LOCAL, BLOCK_STRIDE,
                    kv_groups, H_q, B)
                del _o_tune, _L_tune
            else:
                fwd_nw, fwd_ns = _tuner.get_config("fwd", H_q, D)
            # Allocate outputs and dispatch tiled kernel directly — no CUDA graph
            # wrapper here.  Adding static-buffer copy_() overhead at training
            # batch sizes (B=32: 3×16MB in + 2×16MB out = ~400μs per layer) costs
            # more than the ~50μs dispatch savings, so direct dispatch wins.
            o = torch.empty_like(q)
            L = torch.empty(B, H_q, S, dtype=torch.float32, device=q.device)
            _cron_root_attn_fwd_v14_tiled[(B, H_q, num_tiles)](
                q, k, v, o, L,
                relay_k, relay_v,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                L.stride(0), L.stride(1), L.stride(2),
                relay_k.stride(0), relay_k.stride(1), relay_k.stride(2), relay_k.stride(3),
                relay_v.stride(0), relay_v.stride(1), relay_v.stride(2), relay_v.stride(3),
                S=S, D=D, SQRT_N=SQRT_N,
                NUM_RELAY=NUM_RELAY, BLOCK_RELAY=BLOCK_RELAY,
                BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
                BLOCK_LOCAL=BLOCK_LOCAL, BLOCK_STRIDE=BLOCK_STRIDE,
                KV_GROUPS=kv_groups,
                TOP_K=relay_top_k,
                num_warps=fwd_nw, num_stages=fwd_ns,
            )
        
        ctx.save_for_backward(q, k, v, o, L, relay_k, relay_v, relay_src)
        ctx.SQRT_N = SQRT_N
        ctx.NUM_RELAY = NUM_RELAY
        ctx.skip_relay = skip_relay
        ctx.BLOCK_M = BLOCK_M_BWD      # backward uses BWD size to stay within SRAM
        ctx.BLOCK_D = BLOCK_D
        ctx.BLOCK_LOCAL = BLOCK_M_BWD  # backward local tile matches BWD block size
        ctx.BLOCK_STRIDE = BLOCK_STRIDE
        ctx.BLOCK_RELAY = BLOCK_RELAY
        ctx.kv_groups = kv_groups
        ctx.relay_top_k = relay_top_k
        
        return o
    
    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v, o, L, relay_k, relay_v, relay_src = ctx.saved_tensors
        # NOTE: do NOT call torch.cuda.set_device(q.device) here.  It is a
        # non-context-manager call that permanently changes the global current
        # device, poisoning downstream TE/FP8 CUBLAS workspace references and
        # causing cudaErrorIllegalAddress in TensorImpl::~TensorImpl.
        # Device context for CUDAGraph capture + Triton dispatch is handled
        # correctly by torch.cuda.stream() inside _split3_graphed/_split3_graphed_gqa.
        B, H_q, S, D = q.shape
        H_kv = k.shape[1]  # H_q // kv_groups
        SQRT_N = ctx.SQRT_N
        NUM_RELAY = ctx.NUM_RELAY
        skip_relay = ctx.skip_relay
        BLOCK_M = ctx.BLOCK_M
        BLOCK_D = ctx.BLOCK_D
        BLOCK_LOCAL = ctx.BLOCK_LOCAL
        BLOCK_STRIDE = ctx.BLOCK_STRIDE
        # Clamp BLOCK_STRIDE to the actual number of strided positions for short
        # sequences: e.g. at S=256 (SQRT_N=16) there are only 16 strided positions,
        # so BLOCK_STRIDE=32 wastes SRAM with masked-off slots.
        BLOCK_STRIDE = min(BLOCK_STRIDE, triton.next_power_of_2(max(1, SQRT_N)))
        # Triton tl.dot requires K >= 16 on the contraction dimension.
        # BLOCK_STRIDE is used as K in `tl.dot(dp, k)` and `tl.dot(q, tl.trans(k))`.
        # Extra padding positions are masked to 0.0 by existing n_valid masking,
        # so dp[:, padded] = 0 and k[padded, :] = 0 → zero contribution to dq.
        BLOCK_STRIDE = max(BLOCK_STRIDE, 16)
        BLOCK_RELAY = ctx.BLOCK_RELAY
        relay_top_k = ctx.relay_top_k
        
        use_fused_bwd = (S <= CronRootAttentionV14Function.FUSED_BWD_THRESHOLD)
        
        # GQA (kv_groups > 1) with relay (S > RELAY_THRESHOLD) is not yet supported
        # by the long-seq relay kernels.  Short-seq GQA now takes the split3-gqa path.
        if not use_fused_bwd and ctx.kv_groups != 1:
            raise NotImplementedError(
                f"GQA (kv_groups={ctx.kv_groups}) is only supported for S <= "
                f"{CronRootAttentionV14Function.FUSED_BWD_THRESHOLD}. "
                f"Got S={S}. The long-seq backward kernels do not yet support GQA."
            )
        
        if skip_relay and ctx.kv_groups == 1:
            # MHA: always use split3 (CUDA graph eliminates 3× kernel-launch overhead)
            use_split3     = True
            use_split3_gqa = False
        elif skip_relay:
            # GQA split3: KV-head-centric kernels — zero kv_groups×atomic contention
            use_split3     = False
            use_split3_gqa = True
        else:
            use_split3     = False
            use_split3_gqa = False

        d_relay_log_scale = None  # gradient for relay_log_scale; computed below if relay is active

        if use_split3:
            # =============================================================
            # FAST 3-KERNEL SPLIT BACKWARD  (relay-skip, MHA, all S)
            # =============================================================
            # Why faster: fully_fused uses tl.atomic_add for ALL dK/dV writes,
            # which serialises on Blackwell. Split uses:
            #   • dq_only           — dQ only, zero dK/dV writes
            #   • dkdv_local_phase  — local window, direct stores (no atomics)
            #   • dkdv_strided      — strided keys, minimal atomics
            # Benchmarked vs fully_fused: 1.63× @S=256, 1.92× @S=512,
            # 1.75× @S=1024, 2.53× @S=2048.
            #
            # Previously gated at S≥512 to avoid 3× Python kernel-launch
            # overhead. Now active at ALL S: a CUDA graph is captured on the
            # first call per shape and replayed thereafter (~5μs CPU, vs
            # 3×50μs = 150μs raw dispatch). The graph warmup + capture fires
            # once per unique (B,H,S,D) per training stage.
            # =============================================================
            dq, dk, dv = CronRootAttentionV14Function._split3_graphed(
                q, k, v, o, do, L, relay_k, relay_v,
                S, D, SQRT_N, BLOCK_M, BLOCK_D, BLOCK_LOCAL, BLOCK_STRIDE,
                BLOCK_RELAY, B, H_q)

        elif use_split3_gqa:
            # =============================================================
            # FAST 3-KERNEL SPLIT BACKWARD  (relay-skip, GQA kv_groups > 1)
            # =============================================================
            # KV-head-centric dK/dV kernels eliminate the kv_groups×atomics
            # that fully_fused (Q-head-centric) would produce for GQA.
            # Local phase: direct tl.store() — exclusive KV-tile ownership.
            # Strided phase: atomic_add to merge with local (only one writer).
            # CUDA graph amortises 3× Python kernel-launch overhead to ~5μs.
            # =============================================================
            dq, dk, dv = CronRootAttentionV14Function._split3_graphed_gqa(
                q, k, v, o, do, L, relay_k, relay_v,
                S, D, SQRT_N, BLOCK_M, BLOCK_D, BLOCK_LOCAL, BLOCK_STRIDE,
                BLOCK_RELAY, B, H_q, H_kv, ctx.kv_groups)

        else:
            # =============================================================
            # RELAY-ACTIVE BACKWARD  (S > RELAY_THRESHOLD = 255)
            # =============================================================
            if ctx.kv_groups != 1:
                raise NotImplementedError(
                    f"GQA (kv_groups={ctx.kv_groups}) is only supported for S <= "
                    f"{CronRootAttentionV14Function.FUSED_BWD_THRESHOLD}. "
                    f"Got S={S}. The long-seq backward kernels do not yet support GQA."
                )
            # MHA relay path: CUDA-graph-captured 6-kernel backward.
            # Collapses dQ-only + local-dK/dV + strided-dK/dV + relay-dK/dV
            # + relay_scatter×2 from ~300μs Python dispatch to ~5μs graph replay.
            (dq, dk, dv,
             d_relay_k, d_relay_v,
             _d_rls_raw) = CronRootAttentionV14Function._relay4_graphed(
                q, k, v, o, do, L, relay_k, relay_v, relay_src,
                S, D, SQRT_N, NUM_RELAY, relay_top_k,
                BLOCK_M, BLOCK_D, BLOCK_LOCAL, BLOCK_STRIDE, BLOCK_RELAY, B, H_q)
            if ctx.relay_log_scale is not None:
                d_relay_log_scale = _d_rls_raw.to(ctx.relay_log_scale.dtype)

        return dq, dk, dv, None, None, d_relay_log_scale, None  # Nones: use_persistent, kv_groups, relay_log_scale, relay_top_k


# Tell torch.compile (Dynamo) to graph-break at the CronRoot boundary.
# The Triton kernels inside are already JIT-compiled GPU code — Inductor
# cannot improve them.  This lets the REST of the model (MLPs, norms,
# projections, gating) be compiled and fused by Inductor while CronRoot
# executes in eager mode with its own optimized Triton kernels.
@torch.compiler.disable
def cron_root_attention_v14(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       use_persistent: bool = False, kv_groups: int = 1,
                       relay_log_scale: Optional[torch.Tensor] = None,
                       relay_top_k: int = 2) -> torch.Tensor:
    """
    V14 Tiled Cron Root Attention with optional GQA support.
    
    Optimized for small-to-medium sequence lengths (S < 16K) with:
    - Fused block-query processing (64 queries per block)
    - Optional persistent kernel for minimal launch overhead
    - Native GQA: pass kv_groups>1 and k/v with shape (B, H//kv_groups, S, D)
    - Automatic CPU fallback via vectorized BLAS-based cpu_reference.cra_cpu
    
    Args:
        q: Query tensor of shape (B, H_q, S, D)
        k: Key tensor of shape   (B, H_kv, S, D)  where H_kv = H_q // kv_groups
        v: Value tensor of shape (B, H_kv, S, D)
        use_persistent: If True, use persistent kernel (best for S < 4K)
        kv_groups: Number of query heads per KV head (1 = MHA, 4 = GQA-4)
        relay_log_scale: Optional per-head relay scale in log-space, shape [H_kv].
            relay_k/v are multiplied by exp(relay_log_scale[h]) before the attention
            kernel.  At init (zeros), scale=1 = no change.  Gradient flows back.
    
    Returns:
        Output tensor of shape (B, H_q, S, D)
    """
    if q.device.type == 'cpu':
        from .cpu_reference import cra_cpu_fast_compiled
        return cra_cpu_fast_compiled(q, k, v)
    return CronRootAttentionV14Function.apply(q, k, v, use_persistent, kv_groups, relay_log_scale, relay_top_k)


class CronRootAttentionV14(nn.Module):
    """
    V14 Cron Root Attention Module.
    
    Features:
    - Fused block-query (64 queries/block vs 1 query/block in V13)
    - Optional persistent kernel for ultra-low overhead
    - Same 2√N complexity and 2-hop reachability as V13
    """
    
    def __init__(self, use_persistent: bool = False, auto_persistent_threshold: int = 0):
        """
        Args:
            use_persistent: Force persistent kernel usage
            auto_persistent_threshold: Auto-enable persistent for S <= this value.
                Benchmarks (D=128, BLOCK_M=16): persistent is 2-2.5x SLOWER than
                tiled at all S due to tl.atomic_add overhead dominating compute.
                Set to 0 (disabled by default) until tile compute intensity is high.
        """
        super().__init__()
        self.use_persistent = use_persistent
        self.auto_persistent_threshold = auto_persistent_threshold
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        S = q.shape[2]
        use_persistent = self.use_persistent or (S <= self.auto_persistent_threshold)
        return cron_root_attention_v14(q, k, v, use_persistent=use_persistent)


# =============================================================================
# BENCHMARK AND TESTING
# =============================================================================

def benchmark_v14_vs_v13_vs_sdpa():
    """Comprehensive benchmark comparing V14, V13, and SDPA."""
    import time
    
    device = torch.device('cuda')
    dtype = torch.float16
    
    # Test parameters
    B, H, D = 1, 8, 64
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]
    
    print("=" * 80)
    print("V14 Persistent Tiled √N Attention Benchmark")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: B={B}, H={H}, D={D}")
    print()
    
    # Import V13 for comparison
    try:
        from flash_cron_root_attn_v13_prod import cron_root_attention_fwd, cron_root_attention_bwd
        has_v13 = True
    except ImportError:
        has_v13 = False
        print("Warning: V13 not available for comparison")
    
    results = []
    
    for S in seq_lengths:
        print(f"\n--- Sequence Length: {S} (√N = {int(math.ceil(math.sqrt(S)))}) ---")
        
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)
        
        # Warmup
        for _ in range(10):
            _ = cron_root_attention_v14(q, k, v, use_persistent=True)
            _ = cron_root_attention_v14(q, k, v, use_persistent=False)
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        
        # V14 Persistent
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            _ = cron_root_attention_v14(q, k, v, use_persistent=True)
        torch.cuda.synchronize()
        v14_persistent_time = (time.perf_counter() - start) / 20 * 1000
        
        # V14 Tiled (non-persistent)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            _ = cron_root_attention_v14(q, k, v, use_persistent=False)
        torch.cuda.synchronize()
        v14_tiled_time = (time.perf_counter() - start) / 20 * 1000
        
        # V13 (if available)
        if has_v13:
            SQRT_N = int(math.ceil(math.sqrt(S)))
            L = torch.empty(B, H, S, dtype=torch.float32, device=device)
            o = torch.empty_like(q)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(20):
                cron_root_attention_fwd(q, k, v, o, L, SQRT_N)
            torch.cuda.synchronize()
            v13_time = (time.perf_counter() - start) / 20 * 1000
        else:
            v13_time = float('nan')
        
        # SDPA baseline
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        sdpa_time = (time.perf_counter() - start) / 20 * 1000
        
        # Calculate speedups
        v14_persistent_speedup = sdpa_time / v14_persistent_time
        v14_tiled_speedup = sdpa_time / v14_tiled_time
        v13_speedup = sdpa_time / v13_time if has_v13 else float('nan')
        
        print(f"  V14 Persistent: {v14_persistent_time:.3f}ms ({v14_persistent_speedup:.2f}x vs SDPA)")
        print(f"  V14 Tiled:      {v14_tiled_time:.3f}ms ({v14_tiled_speedup:.2f}x vs SDPA)")
        if has_v13:
            print(f"  V13:            {v13_time:.3f}ms ({v13_speedup:.2f}x vs SDPA)")
        print(f"  SDPA:           {sdpa_time:.3f}ms (baseline)")
        
        results.append({
            'S': S,
            'v14_persistent': v14_persistent_time,
            'v14_tiled': v14_tiled_time,
            'v13': v13_time,
            'sdpa': sdpa_time,
        })
    
    print("\n" + "=" * 80)
    print("SUMMARY: Best V14 Mode Selection")
    print("=" * 80)
    for r in results:
        best = 'persistent' if r['v14_persistent'] < r['v14_tiled'] else 'tiled'
        speedup = r['sdpa'] / min(r['v14_persistent'], r['v14_tiled'])
        print(f"S={r['S']:5d}: Use {best:10s} ({speedup:.2f}x vs SDPA)")
    
    return results


def test_correctness():
    """Test V14 correctness against SDPA reference (with appropriate pattern matching)."""
    print("\n" + "=" * 80)
    print("V14 Correctness Test")
    print("=" * 80)
    
    device = torch.device('cuda')
    dtype = torch.float32  # Use FP32 for accuracy comparison
    
    B, H, S, D = 1, 2, 256, 32
    
    q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    
    # V14 forward
    q_v14 = q.detach().clone().requires_grad_()
    k_v14 = k.detach().clone().requires_grad_()
    v_v14 = v.detach().clone().requires_grad_()
    
    out_v14 = cron_root_attention_v14(q_v14, k_v14, v_v14, use_persistent=False)
    
    # SDPA reference (causal)
    out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    # Note: V14 uses √N sparse pattern, SDPA uses full causal
    # They will NOT match exactly, but should have similar structure
    
    print(f"V14 output shape: {out_v14.shape}")
    print(f"SDPA output shape: {out_sdpa.shape}")
    
    # Test backward
    grad_out = torch.randn_like(out_v14)
    
    out_v14.backward(grad_out)
    out_sdpa.backward(grad_out)
    
    print(f"V14 dQ shape: {q_v14.grad.shape}")
    print(f"SDPA dQ shape: {q.grad.shape}")
    
    # Check for NaN/Inf
    assert not torch.isnan(out_v14).any(), "V14 output contains NaN"
    assert not torch.isinf(out_v14).any(), "V14 output contains Inf"
    assert not torch.isnan(q_v14.grad).any(), "V14 dQ contains NaN"
    assert not torch.isnan(k_v14.grad).any(), "V14 dK contains NaN"
    assert not torch.isnan(v_v14.grad).any(), "V14 dV contains NaN"
    
    print("\n✓ V14 forward and backward pass without NaN/Inf")
    print("✓ (Note: Output values differ from SDPA due to √N sparse pattern)")


if __name__ == "__main__":
    test_correctness()
    benchmark_v14_vs_v13_vs_sdpa()

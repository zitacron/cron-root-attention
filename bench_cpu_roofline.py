"""
CPU Roofline Experiment for CRA
================================
Goal: find a CRA config that is compute-bound (not DRAM/L3-latency-bound)
      on Ryzen 9 7900X running under RAM pressure.

Key insight:
  CRA Phase 2 (strided) AI = 1 FLOP/byte when processing 1 query at a time.
  But when we tile BLOCK_M queries × BLOCK_K strided keys in a matmul:
      AI = 2 * BLOCK_M * BLOCK_K / (BLOCK_M + BLOCK_K)
  For BLOCK_M = BLOCK_K = sqrt(T): AI = sqrt(T)  ← same as the FLOPs ratio!

  At T=4096:  sqrt(T) = 64 FLOP/byte  (L3 ridge ≈ 2, compute-bound ✓)
  At T=256:   sqrt(T) = 16 FLOP/byte  (still compute-bound at L3)

Hardware:
  Ryzen 9 7900X: 12c/24t, AVX2 FP32
  - Theoretical peak: 12 cores × 16 FP32/cycle (AVX2 FMA) × 5.5 GHz ≈ 1056 GFLOP/s
  - L3 bandwidth: ~300 GB/s (2 CCDs)
  - DRAM bandwidth: ~96 GB/s DDR5-6000 (but saturated → ~25 GB/s effective today)
  - L3 RIDGE POINT: 1056 / 300 ≈ 3.5 FLOP/byte
  - DRAM RIDGE POINT: 1056 / 25 ≈ 42 FLOP/byte (with saturation)

L3 cache fit analysis (float32, H=1, d=64):
  T=1024:  SDPA matrix = 1024×1024×4 =  4 MB  (fits L3 → SDPA also fast)
  T=4096:  SDPA matrix = 4096×4096×4 = 64 MB  (hits L3 limit)
  T=8192:  SDPA matrix = 8192×8192×4 = 256 MB (DRAM → SDPA is DRAM-bound)

  CRA hot set = 3*sqrt(T) keys per tile:
  T=8192:  3*91 × 64 × 4 = 70 KB per query tile → always in L1/L2 ✓

SWEET SPOT: T=8192, batch of BLOCK_M=64 queries, H=1, d=64
  → CRA: compute-bound (AI=91 >> ridge=3.5), hot set in L2
  → SDPA: DRAM-bound (256 MB matrix spills L3, DRAM saturated)
"""

import torch
import torch.nn.functional as F
import time
import math
import sys

torch.set_num_threads(12)  # Use all Zen4 cores

def cra_cpu(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            block_m: int = 64) -> torch.Tensor:
    """
    Pure PyTorch CRA on CPU.
    q, k, v: [B, H, T, d]
    Implements Phase 1 (local window sqrt_n) + Phase 2 (strided sqrt_n).
    Relay (Phase 3) omitted — we're studying raw access pattern AI.
    Uses batched matmul tiles of BLOCK_M queries for high arithmetic intensity.
    """
    B, H, T, d = q.shape
    sqrt_n = int(math.isqrt(T))
    scale = 1.0 / math.sqrt(d)
    output = torch.zeros_like(q)

    for b in range(B):
        for h in range(H):
            Q = q[b, h]   # [T, d]
            K = k[b, h]   # [T, d]
            V = v[b, h]   # [T, d]
            O = torch.zeros(T, d, dtype=q.dtype)
            lse = torch.full((T,), float('-inf'))

            # Process BLOCK_M queries at a time for high AI
            for q_start in range(0, T, block_m):
                q_end = min(q_start + block_m, T)
                q_tile = Q[q_start:q_end]          # [BM, d]
                bm = q_end - q_start
                m_i = torch.full((bm,), float('-inf'))
                l_i = torch.zeros(bm)
                acc = torch.zeros(bm, d)

                # ── Phase 1: local window [m - sqrt_n, m] ─────────────────
                # Conservative: grow window from q_start - sqrt_n to q_end
                win_start = max(0, q_start - sqrt_n)
                win_end   = q_end   # inclusive of current tile (causal)
                K_local = K[win_start:win_end]      # [W, d]  contiguous ✓
                V_local = V[win_start:win_end]

                scores = (q_tile @ K_local.T) * scale  # [BM, W]

                # Per-query causal mask within window
                q_pos = torch.arange(q_start, q_end).unsqueeze(1)   # [BM,1]
                k_pos = torch.arange(win_start, win_end).unsqueeze(0) # [1,W]
                local_mask = (k_pos <= q_pos) & (k_pos >= q_pos - sqrt_n + 1)
                scores = scores.masked_fill(~local_mask, float('-inf'))

                # Online softmax update
                block_max = scores.max(dim=1).values
                m_new = torch.maximum(m_i, block_max)
                alpha = torch.exp(m_i - m_new)
                p = torch.exp(scores - m_new.unsqueeze(1))
                p = p * local_mask.float()
                l_i = l_i * alpha + p.sum(dim=1)
                acc = acc * alpha.unsqueeze(1) + p @ V_local
                m_i = m_new

                # ── Phase 2: strided positions 0, sqrt_n, 2*sqrt_n, ... ──
                # before the local window start
                strided_positions = list(range(0, max(0, q_start - sqrt_n + 1), sqrt_n))
                if strided_positions:
                    s_idx = torch.tensor(strided_positions, dtype=torch.long)
                    K_strided = K[s_idx]   # [S, d]  — scattered gather
                    V_strided = V[s_idx]

                    scores_s = (q_tile @ K_strided.T) * scale  # [BM, S]

                    # Causal mask: strided pos must be < q_pos - sqrt_n + 1
                    k_pos_s = s_idx.unsqueeze(0)     # [1, S]
                    local_start_per_q = q_pos - sqrt_n + 1  # [BM, 1]
                    strided_mask = k_pos_s < local_start_per_q
                    scores_s = scores_s.masked_fill(~strided_mask, float('-inf'))

                    block_max_s = scores_s.max(dim=1).values
                    has_valid = block_max_s > float('-inf')
                    m_new2 = torch.where(has_valid, torch.maximum(m_i, block_max_s), m_i)
                    alpha2 = torch.where(m_i == float('-inf'),
                                         torch.where(m_new2 == float('-inf'),
                                                      torch.ones_like(m_i),
                                                      torch.zeros_like(m_i)),
                                         torch.exp(m_i - m_new2))
                    p_s = torch.exp(scores_s - m_new2.unsqueeze(1))
                    p_s = p_s * strided_mask.float()
                    l_i = l_i * alpha2 + p_s.sum(dim=1)
                    acc = acc * alpha2.unsqueeze(1) + p_s @ V_strided
                    m_i = m_new2

                # Normalise and store
                denom = l_i.clamp(min=1e-6).unsqueeze(1)
                O[q_start:q_end] = acc / denom

            output[b, h] = O

    return output


def flops_cra(B, H, T, d):
    """Theoretical FLOPs: 2 phases × 2 ops per token (QK + PV) × counts."""
    sqrt_n = int(math.isqrt(T))
    # Phase1: BM queries × sqrt_n keys per query, 2 matmuls (QK + PV)
    phase1 = B * H * T * sqrt_n * 2 * d * 2
    # Phase2: each query attends to T/sqrt_n strided keys on average
    phase2 = B * H * T * sqrt_n * 2 * d * 2
    return phase1 + phase2


def flops_sdpa(B, H, T, d):
    return B * H * 4 * T * T * d  # standard QK^T + softmax + PV


def working_set_sdpa_mb(H, T, d):
    return H * T * T * 4 / 1e6   # float32 attn matrix


def working_set_cra_hot_kb(d, T):
    """Hot data per query tile on CPU: 3*sqrt(T) vectors."""
    sqrt_n = int(math.isqrt(T))
    return 3 * sqrt_n * d * 4 / 1024


def time_cpu(fn, warmup=2, iters=5):
    for _ in range(warmup):
        out = fn()
    torch.cpu.synchronize() if hasattr(torch.cpu, 'synchronize') else None
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


print("=" * 72)
print("CPU Roofline Experiment — Ryzen 9 7900X")
print("RAM is saturated: DRAM BW ≈ 25 GB/s effective (vs 96 GB/s peak)")
print("L3 = 64 MB,  L3 BW ≈ 300 GB/s,  Peak FP32 ≈ 1056 GFLOP/s (all cores)")
print("L3 ridge point: 3.5 FLOP/byte | DRAM ridge (saturated): 42 FLOP/byte")
print("=" * 72)

print(f"\n{'Config':>30} | {'SDPA ws':>9} | {'CRA hot':>9} | {'CRA AI':>8} | "
      f"{'SDPA bound?':>12} | {'CRA bound?':>12}")
print("-" * 95)

configs = [
    # (H, T, d, block_m)
    (1, 1024,  64,  32),
    (1, 4096,  64,  64),
    (1, 8192,  64,  90),   # sqrt(8192)=90
    (1, 8192,  128, 90),
    (2, 4096,  128, 64),
    (4, 2048,  128, 45),
]

for H, T, d, bm in configs:
    sqrt_n = int(math.isqrt(T))
    sdpa_ws = working_set_sdpa_mb(H, T, d)
    cra_hot = working_set_cra_hot_kb(d, T)
    cra_ai  = sqrt_n  # AI ≈ BLOCK_SIZE when block = sqrt(T)
    sdpa_bound = "DRAM" if sdpa_ws > 64 else ("L3" if sdpa_ws > 1 else "L2")
    cra_bound  = "COMPUTE" if cra_ai > 42 else ("L3" if cra_ai > 3.5 else "L2")
    print(f"  H={H} T={T:5d} d={d:3d} bm={bm:3d} | "
          f"{sdpa_ws:7.1f}MB | {cra_hot:6.0f}KB | {cra_ai:6.0f}F/B | "
          f"{sdpa_bound:>12} | {cra_bound:>12}")

# ─────────────────────────────────────────────────────────────────────────────
# Actual timing on two target configs
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("TIMING  (float32, CPU, 12 threads)")
print("=" * 72)

test_cases = [
    # Label,       B, H, T,    d,   bm
    ("T=1024 d=64",  1, 1, 1024,  64,  32),
    ("T=4096 d=64",  1, 1, 4096,  64,  64),
    ("T=8192 d=64",  1, 1, 8192,  64,  90),
    ("T=8192 d=128", 1, 1, 8192, 128,  90),
]

print(f"\n{'Config':>20} | {'CRA ms':>9} | {'SDPA ms':>9} | "
      f"{'CRA GFLOP/s':>12} | {'SDPA GFLOP/s':>12} | {'Speedup':>8}")
print("-" * 80)

for label, B, H, T, d, bm in test_cases:
    # Allocate (small — max 8192×128×3×4 = 12 MB)
    q = torch.randn(B, H, T, d)
    k = torch.randn(B, H, T, d)
    v = torch.randn(B, H, T, d)

    # CRA
    cra_t = time_cpu(lambda: cra_cpu(q, k, v, block_m=bm), warmup=1, iters=3)
    cra_gflops = flops_cra(B, H, T, d) / cra_t / 1e9

    # SDPA (PyTorch CPU — materialises full T×T)
    if T <= 8192:   # skip too-large for available RAM
        sdpa_t = time_cpu(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
            warmup=1, iters=3)
        sdpa_gflops = flops_sdpa(B, H, T, d) / sdpa_t / 1e9
        speedup = sdpa_t / cra_t
        print(f"{label:>20} | {cra_t*1000:>7.1f}ms | {sdpa_t*1000:>7.1f}ms | "
              f"{cra_gflops:>10.1f}G | {sdpa_gflops:>10.1f}G | {speedup:>6.2f}x")
    else:
        print(f"{label:>20} | {cra_t*1000:>7.1f}ms | {'N/A':>9} | "
              f"{cra_gflops:>10.1f}G | {'N/A':>12} | {'N/A':>8}")

    del q, k, v

print("\nNotes:")
print("  CRA GFLOP/s > 42 FLOP/byte × BW = compute-bound (even with saturated RAM)")
print("  SDPA GFLOP/s at large T will be low (DRAM-bound, matrix spills L3)")
print("  Peak single-thread FP32: ~88 GFLOP/s  |  All-core: ~1056 GFLOP/s")

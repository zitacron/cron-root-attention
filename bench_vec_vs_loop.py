"""
bench_vec_vs_loop.py — compare:
  cra_cpu      : loop + pre-compute scores_23 (current default)
  cra_cpu_fast : loop + in-loop P23 QK (eliminates 134 MB pre-compute tensor)
  cra_cpu_vec  : unfold-batched Phase-1 (no loop) — for reference

Run:
  python bench_vec_vs_loop.py              # eager only
  python bench_vec_vs_loop.py --compiled   # also compiled variants
"""
import sys
import os
import time
import math
import torch

sys.path.insert(0, os.path.dirname(__file__))
from cron_root_attention.cpu_reference import cra_cpu, cra_cpu_fast, cra_cpu_vec

# (vec retained for correctness cross-check only)

# ── helpers ───────────────────────────────────────────────────────────────────
torch.set_num_threads(12)

def make_qkv(B, H, T, d, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(); g.manual_seed(seed)
    q = torch.randn(B, H, T, d, generator=g, dtype=dtype)
    k = torch.randn(B, H, T, d, generator=g, dtype=dtype)
    v = torch.randn(B, H, T, d, generator=g, dtype=dtype)
    return q, k, v

def bench(fn, *args, warmup=3, iters=10, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    return (time.perf_counter() - t0) / iters * 1000      # ms

# ── correctness check ─────────────────────────────────────────────────────────
print("=" * 70)
print("CORRECTNESS  cra_cpu_fast vs cra_cpu  (FP32, various shapes)")
for T, d, H in [(256, 32, 4), (512, 64, 4), (1024, 128, 8), (4096, 128, 8)]:
    q, k, v = make_qkv(1, H, T, d, dtype=torch.float32)
    ref = cra_cpu(q, k, v)
    fast = cra_cpu_fast(q, k, v)
    diff = (ref - fast).abs().max().item()
    status = "PASS" if diff < 1e-4 else "FAIL"
    print(f"  T={T:<5} H={H} d={d}  max_diff={diff:.2e}  {status}")

print()
print("CORRECTNESS  cra_cpu_vec vs cra_cpu (for completeness)")
for T, d, H in [(256, 32, 4), (1024, 128, 8)]:
    q, k, v = make_qkv(1, H, T, d, dtype=torch.float32)
    ref = cra_cpu(q, k, v)
    vec = cra_cpu_vec(q, k, v)
    diff = (ref - vec).abs().max().item()
    status = "PASS" if diff < 1e-4 else "FAIL"
    print(f"  T={T:<5} H={H} d={d}  max_diff={diff:.2e}  {status}")

# ── speed benchmark ───────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"SPEED (eager)  B=1, H=16, d=128, BF16, chunk=256, OMP=12")
print(f"{'T':>6}  {'loop ms':>9}  {'fast ms':>9}  {'fast/loop':>10}")
print("-" * 55)

B, H, d = 1, 16, 128
for T in [2048, 4096, 8192, 16384]:
    q, k, v = make_qkv(B, H, T, d, dtype=torch.bfloat16)
    t_loop = bench(cra_cpu,      q, k, v, warmup=2, iters=6)
    t_fast = bench(cra_cpu_fast, q, k, v, warmup=2, iters=6)
    flag = "  <-- BETTER" if t_fast < t_loop else ""
    print(f"{T:>6}  {t_loop:>9.1f}  {t_fast:>9.1f}  {t_loop/t_fast:>9.2f}×{flag}")

if "--compiled" in sys.argv:
    from cron_root_attention.cpu_reference import (
        cra_cpu_compiled, cra_cpu_fast_compiled
    )

    print()
    print("=" * 70)
    print("Compiling all three variants (first call triggers compile ~30s each)...")
    q0, k0, v0 = make_qkv(1, 16, 4096, 128, dtype=torch.bfloat16)
    print("  compiling cra_cpu_compiled ...", flush=True)
    _ = cra_cpu_compiled(q0, k0, v0)
    print("  compiling cra_cpu_fast_compiled ...", flush=True)
    _ = cra_cpu_fast_compiled(q0, k0, v0)
    print("Done.\n")

    print(f"SPEED (compiled)  B=1, H=16, d=128, BF16, chunk=256, OMP=12")
    print(f"{'T':>6}  {'loop-c ms':>10}  {'fast-c ms':>10}  {'fast/loop':>10}")
    print("-" * 55)
    for T in [4096, 8192, 16384]:
        q, k, v = make_qkv(1, 16, T, 128, dtype=torch.bfloat16)
        t_lc  = bench(cra_cpu_compiled,      q, k, v, warmup=3, iters=8)
        t_fc  = bench(cra_cpu_fast_compiled, q, k, v, warmup=3, iters=8)
        flag  = "  <-- BETTER" if t_fc < t_lc else ""
        print(f"{T:>6}  {t_lc:>10.1f}  {t_fc:>10.1f}  {t_lc/t_fc:>9.2f}×{flag}")

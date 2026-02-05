"""
Cron Root Attention - Full Benchmark Suite
===========================================

Benchmarks Cron Root Attention vs SDPA (FlashAttention-2 backend)
and SDPA (math/memory-efficient backend) across:
  1. Forward pass only (kernel benchmark)
  2. Forward + Backward (training benchmark)
  3. Inference (no_grad, simulating generation)

Uses CUDA events for precise GPU timing.
"""

import torch
import torch.nn.functional as F
import math
import json
import sys
import gc

from cron_root_attention import cron_root_attention, get_gpu_info


def cuda_timer(func, warmup=10, repeats=30):
    """Time a GPU function using CUDA events (most accurate)."""
    # Warmup
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(repeats):
        start_event.record()
        func()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))  # ms

    times.sort()
    # Trim top/bottom 10% for stable median
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim] if len(times) > 2 * trim else times
    return sum(trimmed) / len(trimmed)


def benchmark_forward(B, H, D, seq_lengths, dtype=torch.float16):
    """Benchmark forward pass only."""
    device = torch.device('cuda')
    results = []

    for S in seq_lengths:
        gc.collect()
        torch.cuda.empty_cache()

        try:
            q = torch.randn(B, H, S, D, device=device, dtype=dtype)
            k = torch.randn(B, H, S, D, device=device, dtype=dtype)
            v = torch.randn(B, H, S, D, device=device, dtype=dtype)
        except torch.cuda.OutOfMemoryError:
            print(f"  S={S}: OOM on allocation, skipping")
            break

        row = {"seq_len": S}

        # --- Cron Root Attention ---
        try:
            cron_ms = cuda_timer(lambda: cron_root_attention(q, k, v))
            row["cron_root_ms"] = round(cron_ms, 4)
        except Exception as e:
            print(f"  S={S} CronRoot forward error: {e}")
            row["cron_root_ms"] = None

        # --- SDPA (Flash backend) ---
        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_mem_efficient=False, enable_math=False
            ):
                sdpa_flash_ms = cuda_timer(
                    lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)
                )
            row["sdpa_flash_ms"] = round(sdpa_flash_ms, 4)
        except Exception:
            # Flash may not support this config, fall through
            row["sdpa_flash_ms"] = None

        # --- SDPA (mem-efficient backend) ---
        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=True, enable_math=False
            ):
                sdpa_mem_ms = cuda_timer(
                    lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)
                )
            row["sdpa_mem_ms"] = round(sdpa_mem_ms, 4)
        except Exception:
            row["sdpa_mem_ms"] = None

        # --- SDPA (auto - best available) ---
        try:
            sdpa_auto_ms = cuda_timer(
                lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)
            )
            row["sdpa_auto_ms"] = round(sdpa_auto_ms, 4)
        except Exception:
            row["sdpa_auto_ms"] = None

        # Speedups
        best_sdpa = row.get("sdpa_auto_ms") or row.get("sdpa_flash_ms") or row.get("sdpa_mem_ms")
        if best_sdpa and row["cron_root_ms"]:
            row["speedup_vs_sdpa"] = round(best_sdpa / row["cron_root_ms"], 2)

        results.append(row)
        print(f"  S={S:>7}: Cron={row.get('cron_root_ms','OOM'):>8}ms  "
              f"SDPA(flash)={row.get('sdpa_flash_ms','N/A'):>8}ms  "
              f"SDPA(auto)={row.get('sdpa_auto_ms','N/A'):>8}ms  "
              f"Speedup={row.get('speedup_vs_sdpa','N/A')}x")

        del q, k, v
        torch.cuda.empty_cache()

    return results


def benchmark_training(B, H, D, seq_lengths, dtype=torch.float16):
    """Benchmark forward + backward pass (training)."""
    device = torch.device('cuda')
    results = []

    for S in seq_lengths:
        gc.collect()
        torch.cuda.empty_cache()

        try:
            q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  S={S}: OOM on allocation, skipping")
            break

        row = {"seq_len": S}

        # --- Cron Root: fwd + bwd ---
        def cron_fwd_bwd():
            q.grad = k.grad = v.grad = None
            o = cron_root_attention(q, k, v)
            loss = o.sum()
            loss.backward()

        try:
            cron_ms = cuda_timer(cron_fwd_bwd, warmup=5, repeats=20)
            row["cron_root_ms"] = round(cron_ms, 4)
        except Exception as e:
            print(f"  S={S} CronRoot train error: {e}")
            row["cron_root_ms"] = None

        # --- SDPA (auto): fwd + bwd ---
        def sdpa_fwd_bwd():
            q.grad = k.grad = v.grad = None
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            loss = o.sum()
            loss.backward()

        try:
            sdpa_ms = cuda_timer(sdpa_fwd_bwd, warmup=5, repeats=20)
            row["sdpa_auto_ms"] = round(sdpa_ms, 4)
        except Exception as e:
            print(f"  S={S} SDPA train error: {e}")
            row["sdpa_auto_ms"] = None

        # --- SDPA (flash only): fwd + bwd ---
        def sdpa_flash_fwd_bwd():
            q.grad = k.grad = v.grad = None
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_mem_efficient=False, enable_math=False
            ):
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            loss = o.sum()
            loss.backward()

        try:
            sdpa_flash_ms = cuda_timer(sdpa_flash_fwd_bwd, warmup=5, repeats=20)
            row["sdpa_flash_ms"] = round(sdpa_flash_ms, 4)
        except Exception:
            row["sdpa_flash_ms"] = None

        # Speedups
        best_sdpa = row.get("sdpa_auto_ms")
        if best_sdpa and row.get("cron_root_ms"):
            row["speedup_vs_sdpa"] = round(best_sdpa / row["cron_root_ms"], 2)

        results.append(row)
        print(f"  S={S:>7}: Cron={row.get('cron_root_ms','OOM'):>8}ms  "
              f"SDPA(flash)={row.get('sdpa_flash_ms','N/A'):>8}ms  "
              f"SDPA(auto)={row.get('sdpa_auto_ms','N/A'):>8}ms  "
              f"Training speedup={row.get('speedup_vs_sdpa','N/A')}x")

        del q, k, v
        torch.cuda.empty_cache()

    return results


def benchmark_inference(B, H, D, seq_lengths, dtype=torch.float16):
    """Benchmark inference (no_grad, simulating prefill)."""
    device = torch.device('cuda')
    results = []

    for S in seq_lengths:
        gc.collect()
        torch.cuda.empty_cache()

        try:
            q = torch.randn(B, H, S, D, device=device, dtype=dtype)
            k = torch.randn(B, H, S, D, device=device, dtype=dtype)
            v = torch.randn(B, H, S, D, device=device, dtype=dtype)
        except torch.cuda.OutOfMemoryError:
            print(f"  S={S}: OOM on allocation, skipping")
            break

        row = {"seq_len": S}

        # --- Cron Root inference ---
        try:
            with torch.no_grad():
                cron_ms = cuda_timer(lambda: cron_root_attention(q, k, v))
            row["cron_root_ms"] = round(cron_ms, 4)
        except Exception as e:
            print(f"  S={S} CronRoot inference error: {e}")
            row["cron_root_ms"] = None

        # --- SDPA (auto) inference ---
        try:
            with torch.no_grad():
                sdpa_ms = cuda_timer(
                    lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)
                )
            row["sdpa_auto_ms"] = round(sdpa_ms, 4)
        except Exception:
            row["sdpa_auto_ms"] = None

        # --- SDPA (flash only) inference ---
        try:
            with torch.no_grad(), torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_mem_efficient=False, enable_math=False
            ):
                sdpa_flash_ms = cuda_timer(
                    lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)
                )
            row["sdpa_flash_ms"] = round(sdpa_flash_ms, 4)
        except Exception:
            row["sdpa_flash_ms"] = None

        # Speedups
        best_sdpa = row.get("sdpa_auto_ms")
        if best_sdpa and row.get("cron_root_ms"):
            row["speedup_vs_sdpa"] = round(best_sdpa / row["cron_root_ms"], 2)

        results.append(row)
        print(f"  S={S:>7}: Cron={row.get('cron_root_ms','OOM'):>8}ms  "
              f"SDPA(flash)={row.get('sdpa_flash_ms','N/A'):>8}ms  "
              f"SDPA(auto)={row.get('sdpa_auto_ms','N/A'):>8}ms  "
              f"Inference speedup={row.get('speedup_vs_sdpa','N/A')}x")

        del q, k, v
        torch.cuda.empty_cache()

    return results


def main():
    gpu_info = get_gpu_info()
    print("=" * 80)
    print("Cron Root Attention - Full Benchmark Suite")
    print("=" * 80)
    print(f"GPU: {gpu_info['gpu_name']} ({gpu_info['sm_count']} SMs)")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    B, H, D = 1, 8, 64

    # Sequence lengths to test
    fwd_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
    train_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    infer_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]

    print(f"Config: B={B}, H={H}, D={D}, dtype=float16")
    print()

    # ---- Forward Pass ----
    print("─" * 80)
    print("1. FORWARD PASS (Kernel Only)")
    print("─" * 80)
    fwd_results = benchmark_forward(B, H, D, fwd_lengths)
    print()

    # ---- Training (Fwd + Bwd) ----
    print("─" * 80)
    print("2. TRAINING (Forward + Backward)")
    print("─" * 80)
    train_results = benchmark_training(B, H, D, train_lengths)
    print()

    # ---- Inference ----
    print("─" * 80)
    print("3. INFERENCE (no_grad prefill)")
    print("─" * 80)
    infer_results = benchmark_inference(B, H, D, infer_lengths)
    print()

    # ---- Summary JSON ----
    all_results = {
        "gpu": gpu_info['gpu_name'],
        "sm_count": gpu_info['sm_count'],
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "config": {"B": B, "H": H, "D": D, "dtype": "float16"},
        "forward": fwd_results,
        "training": train_results,
        "inference": infer_results,
    }

    with open("/tmp/benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("=" * 80)
    print("Results saved to /tmp/benchmark_results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
Cron Root Attention Benchmark
==============================

Compare Cron Root Attention vs SDPA at various sequence lengths.
"""

import torch
import torch.nn.functional as F
import time
import math

from cron_root_attention import cron_root_attention, get_gpu_info


def benchmark():
    """Run comprehensive benchmark."""
    device = torch.device('cuda')
    dtype = torch.float16
    
    # Test parameters
    B, H, D = 1, 8, 64
    seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    
    print("=" * 80)
    print("Cron Root Attention Benchmark")
    print("=" * 80)
    
    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info['gpu_name']} ({gpu_info['sm_count']} SMs)")
    print(f"Config: B={B}, H={H}, D={D}")
    print()
    
    print(f"{'Seq Len':>10} | {'Cron Root':>12} | {'SDPA':>12} | {'Speedup':>10}")
    print("-" * 52)
    
    for S in seq_lengths:
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)
        
        # Warmup
        for _ in range(5):
            _ = cron_root_attention(q, k, v)
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        
        # Benchmark Cron Root
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            _ = cron_root_attention(q, k, v)
        torch.cuda.synchronize()
        cron_time = (time.perf_counter() - start) / 20 * 1000
        
        # Benchmark SDPA
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        sdpa_time = (time.perf_counter() - start) / 20 * 1000
        
        speedup = sdpa_time / cron_time
        
        print(f"{S:>10} | {cron_time:>10.3f}ms | {sdpa_time:>10.3f}ms | {speedup:>8.1f}x")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    benchmark()

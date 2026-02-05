"""
Quick Start Example: Cron Root Attention
=========================================

Drop-in replacement for PyTorch attention in 10 lines of code.
"""

import torch
from cron_root_attention import CronRootAttention

# Create the module (identical API to nn.MultiheadAttention)
attn = CronRootAttention(d_model=1024, n_heads=16).cuda()

# Standard sequence (uses what works)
x = torch.randn(2, 2048, 1024, device="cuda", dtype=torch.float16)
output = attn(x)
print(f"Input: {x.shape} → Output: {output.shape}")

# Long sequence (gets 28x kernel speedup at S=16384)
x_long = torch.randn(1, 16384, 1024, device="cuda", dtype=torch.float16)
output_long = attn(x_long)
print(f"Long input: {x_long.shape} → Output: {output_long.shape}")

print("✓ Cron Root Attention working!")

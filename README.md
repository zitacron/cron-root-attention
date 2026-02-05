# Cron Root Attention™

**Sub-quadratic O(N√N) attention with 2-hop relay for long-context transformers**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg)](https://pytorch.org/)

## Key Results

Cron Root Attention achieves **up to 202x kernel speedups** over standard SDPA/FlashAttention at long sequence lengths by reducing attention complexity from O(N²) to O(N√N), with **100% token coverage** through a 3-phase relay mechanism.

### Forward Pass Benchmarks (Kernel Only)

| Sequence Length | Cron Root | SDPA | Speedup |
|-----------------|-----------|------|---------|
| 1,024 | 0.03ms | 0.06ms | **2.0x** |
| 4,096 | 0.04ms | 0.50ms | **12.7x** |
| 16,384 | 0.22ms | 6.33ms | **28.2x** |
| 32,768 | 0.59ms | 24.1ms | **40.6x** |
| 65,536 | 0.77ms | 50.9ms | **66.1x** |
| 131,072 | 2.08ms | 203ms | **97.6x** |
| 262,144 | 5.47ms | 809ms | **148x** |
| 524,288 | 15.1ms | 3050ms | **202x** |

*Benchmarked on RTX 5070 Ti (Blackwell GB203), FP16, B=1, H=8, D=64*

### End-to-End Training Performance (Forward + Backward)

| Sequence Length | Cron Root Fwd | Cron Root Bwd | SDPA Total | Training Speedup |
|-----------------|---------------|---------------|------------|------------------|
| 4,096 | 0.053ms | 0.71ms | 0.94ms | **1.20x** |
| 8,192 | 0.079ms | 1.74ms | 2.95ms | **1.63x** |
| 16,384 | 0.168ms | 4.51ms | 11.21ms | **2.49x** |
| 32,768 | 0.331ms | 12.04ms | 43.17ms | **3.38x** |
| 65,536 | 0.771ms | 32.52ms | 168.25ms | **5.05x** |
| 131,072 | 2.077ms | 91.95ms | 670.93ms | **7.16x** |

The training speedup uses our **key-centric backward pass** which eliminates atomic contention.

> **Note**: Attention is ~30-40% of total training compute. The remaining FFN, LayerNorm, and embedding operations limit the theoretical maximum speedup per Amdahl's Law. For **inference-only** workloads, the full 202x kernel speedup applies.

## 📦 Installation

```bash
pip install cron-root-attention
```

Or from source:
```bash
git clone https://github.com/zitacron/cron-root-attention.git
cd cron-root-attention
pip install -e .
```

## ⚡ Quick Start

### 10-Line Integration

```python
import torch
from cron_root_attention import cron_root_attention

# Your existing Q, K, V tensors (B, H, S, D)
q = torch.randn(1, 16, 8192, 64, device='cuda', dtype=torch.float16)
k = torch.randn(1, 16, 8192, 64, device='cuda', dtype=torch.float16)
v = torch.randn(1, 16, 8192, 64, device='cuda', dtype=torch.float16)

# Drop-in replacement for scaled_dot_product_attention
output = cron_root_attention(q, k, v)  # 14x faster at S=8192!
```

### Module API (Drop-in for nn.MultiheadAttention)

```python
from cron_root_attention import CronRootMultiheadAttention

# Replace nn.MultiheadAttention
attn = CronRootMultiheadAttention(
    embed_dim=1024,
    num_heads=16,
    batch_first=True
).cuda()

x = torch.randn(1, 8192, 1024, device='cuda', dtype=torch.float16)
output, _ = attn(x, x, x)  # Automatic √N sparse attention
```

### Hybrid Mode (Auto Backend Selection)

Because of the sub quadratic complexity nature of this attention mechanisim, smaller than 1024 or 512 usually results in less performance than the standard quadratic attention.
So this is to auto-select attention to get the max speedup possible.

```python
from cron_root_attention import cron_root_attention_hybrid

# Automatically uses SDPA for S < 1024, Cron Root for S >= 1024
output = cron_root_attention_hybrid(q, k, v, backend='auto')
```

## How It Works

### The 3-Phase √N Sparse Attention Pattern

Instead of attending to all N previous tokens (O(N²)), each query attends to 3√N tokens across three phases:

1. **Phase 1 — Local Window (√N tokens)**: Immediate predecessors for fine-grained context
2. **Phase 2 — Strided Window (√N tokens)**: Every √N-th token for global sampling
3. **Phase 3 — Relay Keys (√N blocks)**: Block-mean summaries carrying compressed 2-hop information

```
Query at position 100 (√N = 23, S = 512):

  Phase 1 - Local:   [78, 79, ..., 100]              (23 tokens, exact)
  Phase 2 - Strided: [0, 23, 46, 69]                 (4 tokens, exact, before local)
  Phase 3 - Relay:   [block0_mean, block1_mean, ...]  (≤23 compressed blocks)
                      each block summarizes 23 tokens

  Softmax slots: 3√N = 69
  Token coverage: 512/512 = 100% (relay blocks cover entire sequence)
```

### 2-Hop Relay Mechanism

The relay mechanism solves the **gradient dilution problem** inherent in multi-hop sparse attention. Without relay, 2-hop information must survive two separate softmax normalizations across layers — gradients wash out exponentially.

**Relay solves this by carrying compressed 2-hop information through a single softmax:**

```
Pre-computation (PyTorch, before kernel launch):
  relay_k[r] = mean(K[r·√N : (r+1)·√N])   # Block-mean keys
  relay_v[r] = mean(V[r·√N : (r+1)·√N])   # Block-mean values
  Shape: (B, H, NUM_RELAY, D) where NUM_RELAY ≈ √N

Inside the kernel (same softmax as Phase 1 & 2):
  For each query q_m:
    score(q_m, relay_k[r]) participates in the SAME online softmax
    as local and strided scores — single normalization, no dilution

Backward (gradient scatter via chain rule):
  dK[r·√N + i] += d_relay_k[r] / √N   for all i in block r
```

**Result**: Every query sees the entire sequence — local positions exactly, strided positions exactly, and all remaining positions through compressed relay blocks. All in O(N√N) complexity.

### Key-Centric Backward Pass

The backward pass uses a novel key-centric strided kernel:
- Each block **owns** one strided key and iterates over all queries
- Zero atomic contention (vs O(N) atomics in query-parallel)
- Register accumulation → single write at end
- **1.85x speedup** in strided phase (21.95ms → 11.85ms at S=64K)

The relay backward uses the same exclusive-ownership pattern:
- Each block owns one relay key/value pair
- Iterates over all queries that attend to it
- Zero atomics, register accumulation
- Gradient scatter: `dK[r·√N+i] += d_relay_k[r] / √N`

## 📊 Benchmarks

### Complexity Comparison

| Method | Time Complexity | Memory | Pattern | Full Coverage |
|--------|----------------|--------|---------|---------------|
| Dense SDPA | O(N²) | O(N) | Full causal | Yes |
| FlashAttention-2 | O(N²) | O(N) | Full causal | Yes |
| Longformer | O(N·w) | O(N) | Local + global | Limited |
| **Cron Root (Ours)** | **O(N√N)** | **O(N)** | **Local + Strided + Relay** | **Yes (2-hop)** |

### Supported Hardware

Automatic SM detection supports 40+ GPU models:

| Category | GPUs |
|----------|------|
| Blackwell (50 series) | RTX 5090, 5080, 5070 Ti, 5070 |
| Ada (40 series) | RTX 4090, 4080, 4070 Ti, 4070, 4060 Ti |
| Ampere (30 series) | RTX 3090, 3080, 3070, 3060 |
| Turing (20 series) | RTX 2080 Ti, 2080, 2070, 2060, TITAN RTX |
| Datacenter | H100, H200, H800, A100, L40S, L4, V100, B100, B200 |

```python
from cron_root_attention import get_gpu_info
print(get_gpu_info())
# {'gpu_name': 'NVIDIA GeForce RTX 5070 Ti', 'sm_count': 70, 'is_known_gpu': True}
```

## Requirements

- Python 3.10+
- PyTorch 2.2+
- Triton 2.2+
- CUDA 12.0+ (Blackwell/Hopper recommended)

## 📄 Citation

If you use Cron Root Attention in your research, please cite:
```bibtex
@software{cron_root_attention,
  author = {{Zitacron Project}},
  title = {Cron Root Attention: Sub-quadratic Attention for Long-Context Transformers},
  year = {2026},
  url = {https://github.com/zitacron/cron-root-attention},
  version = {0.1.0},
  note = {Zitacron™ and Cron Root Attention™ are trademarks of the Zitacron Project.}
}
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
NOTICE - See [NOTICE](NOTICE) for further details.

## 🙏 Acknowledgments

- Inspired by [FlashAttention](https://github.com/Dao-AILab/flash-attention) by Tri Dao
- Built on [Triton](https://github.com/openai/triton) by OpenAI
- Optimized for NVIDIA Blackwell architecture (GB203) (more to come!)

**Zitacron** - Building the future of efficient AI

---

© 2026 Zitacron. "Zitacron" and "Cron Root Attention" are trademarks of the Zitacron Project. 
Optimized for NVIDIA Blackwell Architecture.



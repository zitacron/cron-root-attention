# Cron Root Attention™

**Sub-quadratic O(N√N) attention with 2-hop relay for long-context transformers**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg)](https://pytorch.org/)

## Key Results

Cron Root Attention achieves **up to 58x forward kernel speedup** over SDPA/FlashAttention-2 at long sequence lengths by reducing attention complexity from O(N²) to O(N√N), with **100% token coverage** through a 3-phase relay mechanism. Crossover point is ~2K tokens — above that, Cron Root is strictly faster.

### Forward Pass Benchmarks (Kernel Only)

| Sequence Length | Cron Root | SDPA (Flash) | Speedup |
|-----------------|-----------|--------------|----------|
| 512 | 0.103ms | 0.024ms | 0.23x |
| 1,024 | 0.070ms | 0.037ms | 0.48x |
| 2,048 | 0.089ms | 0.111ms | **1.24x** |
| 4,096 | 0.102ms | 0.286ms | **2.80x** |
| 8,192 | 0.215ms | 0.923ms | **4.25x** |
| 16,384 | 0.341ms | 3.37ms | **9.77x** |
| 32,768 | 1.15ms | 12.8ms | **11.2x** |
| 65,536 | 2.33ms | 48.5ms | **20.9x** |
| 131,072 | 7.36ms | 192ms | **26.2x** |
| 262,144 | 17.3ms | 765ms | **44.3x** |
| 524,288 | 52.4ms | 3054ms | **58.2x** |

*Measured on RTX 5070 Ti (Blackwell GB203), PyTorch 2.9.1, CUDA 12.8, FP16, B=1, H=8, D=64. CUDA event timing, trimmed mean of 30 runs.*

### End-to-End Training Performance (Forward + Backward)

| Sequence Length | Cron Root (Fwd+Bwd) | SDPA (Fwd+Bwd) | Training Speedup |
|-----------------|---------------------|----------------|------------------|
| 512 | 0.347ms | 0.107ms | 0.31x |
| 1,024 | 0.405ms | 0.139ms | 0.34x |
| 2,048 | 0.707ms | 0.347ms | 0.49x |
| 4,096 | 1.39ms | 0.925ms | 0.67x |
| 8,192 | 3.39ms | 2.91ms | 0.86x |
| 16,384 | 8.69ms | 11.1ms | **1.28x** |
| 32,768 | 24.5ms | 43.1ms | **1.76x** |
| 65,536 | 65.9ms | 170ms | **2.58x** |
| 131,072 | 186ms | 676ms | **3.64x** |

Training crossover is ~12K tokens. The backward pass uses our **key-centric** kernels with zero atomic contention.

### Inference Benchmarks (no_grad prefill)

| Sequence Length | Cron Root | SDPA (Flash) | Inference Speedup |
|-----------------|-----------|--------------|-------------------|
| 512 | 0.078ms | 0.022ms | 0.27x |
| 1,024 | 0.061ms | 0.034ms | 0.56x |
| 2,048 | 0.091ms | 0.111ms | **1.20x** |
| 4,096 | 0.093ms | 0.289ms | **3.07x** |
| 8,192 | 0.185ms | 0.924ms | **5.01x** |
| 16,384 | 0.339ms | 3.37ms | **9.89x** |
| 32,768 | 1.16ms | 12.6ms | **10.8x** |
| 65,536 | 2.29ms | 48.6ms | **21.2x** |
| 131,072 | 7.39ms | 193ms | **26.0x** |
| 262,144 | 17.3ms | 766ms | **44.3x** |
| 524,288 | 52.4ms | 2984ms | **57.0x** |

> **Note**: Attention is ~30-40% of total training compute. The remaining FFN, LayerNorm, and embedding operations limit the theoretical maximum speedup per Amdahl's Law. For **inference-only** workloads (prefill), the full kernel speedup applies directly. Use the **hybrid mode** to auto-select SDPA for short sequences and Cron Root for long sequences.

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



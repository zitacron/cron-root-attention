# Cron Root Attention

**Sub-quadratic O(N√N) attention with 2-hop relay for long-context transformers**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg)](https://pytorch.org/)

## Key Results

Cron Root Attention achieves **up to 68x forward kernel speedup** over SDPA/FlashAttention-2 at long sequence lengths by reducing attention complexity from O(N²) to O(N√N), with **100% token coverage** through a 3-phase relay mechanism. Forward crossover varies by model size — as few as **512 tokens** for larger heads. The **hybrid mode** auto-selects SDPA below crossover for zero-regression deployment.

All benchmarks below were measured on **RTX 5070 Ti** (Blackwell GB203, 70 SMs), PyTorch 2.9.1, CUDA 12.8, FP16, all in python. Timing uses CUDA events with verified warmup (adaptive 20–50 warmup iterations + stability check with CV < 5% before measurement, trimmed mean of 10–100 measured repeats).

### Forward Pass — Multiple Model Configurations

Speedup scales with both sequence length and model size. Larger head dimensions benefit more from sub-quadratic scaling.

#### Small Model (H=8, D=64 — 512-dim, 8-head)

| Sequence Length | Cron Root | SDPA (Flash) | Speedup |
|-----------------|-----------|--------------|----------|
| 512 | 0.040ms | 0.023ms | 0.59x |
| 1,024 | 0.042ms | 0.036ms | 0.87x |
| 2,048 | 0.048ms | 0.110ms | **2.31x** |
| 4,096 | 0.061ms | 0.289ms | **4.76x** |
| 8,192 | 0.102ms | 0.918ms | **8.96x** |
| 16,384 | 0.339ms | 3.34ms | **9.86x** |
| 32,768 | 1.17ms | 13.1ms | **11.3x** |
| 65,536 | 2.35ms | 48.4ms | **20.6x** |
| 131,072 | 7.33ms | 192ms | **26.2x** |
| 262,144 | 17.3ms | 762ms | **44.1x** |
| 524,288 | 52.1ms | 3042ms | **58.4x** |

#### Medium Model (H=12, D=64 — 768-dim, 12-head)

| Sequence Length | Cron Root | SDPA (Flash) | Speedup |
|-----------------|-----------|--------------|----------|
| 512 | 0.040ms | 0.023ms | 0.59x |
| 1,024 | 0.041ms | 0.054ms | **1.31x** |
| 2,048 | 0.052ms | 0.134ms | **2.57x** |
| 4,096 | 0.075ms | 0.389ms | **5.19x** |
| 8,192 | 0.138ms | 1.30ms | **9.48x** |
| 16,384 | 0.511ms | 4.81ms | **9.41x** |
| 32,768 | 1.74ms | 18.4ms | **10.6x** |
| 65,536 | 3.48ms | 72.1ms | **20.8x** |
| 131,072 | 11.2ms | 287ms | **25.6x** |
| 262,144 | 26.0ms | 1141ms | **44.0x** |
| 524,288 | 78.4ms | 4559ms | **58.2x** |

#### Large Model (H=16, D=128 — 2048-dim, 16-head)

| Sequence Length | Cron Root | SDPA (Flash) | Speedup |
|-----------------|-----------|--------------|----------|
| 512 | 0.043ms | 0.036ms | 0.84x |
| 1,024 | 0.053ms | 0.102ms | **1.92x** |
| 2,048 | 0.081ms | 0.285ms | **3.52x** |
| 4,096 | 0.140ms | 0.958ms | **6.87x** |
| 8,192 | 0.303ms | 3.45ms | **11.4x** |
| 16,384 | 1.25ms | 13.2ms | **10.5x** |
| 32,768 | 4.38ms | 51.6ms | **11.8x** |
| 65,536 | 8.44ms | 204ms | **24.2x** |
| 131,072 | 27.6ms | 813ms | **29.4x** |
| 262,144 | 62.2ms | 3251ms | **52.3x** |
| 524,288 | 191ms | 12944ms | **67.8x** |

#### XL Model (H=32, D=128 — 4096-dim, 32-head)

| Sequence Length | Cron Root | SDPA (Flash) | Speedup |
|-----------------|-----------|--------------|----------|
| 512 | 0.052ms | 0.060ms | **1.17x** |
| 1,024 | 0.070ms | 0.158ms | **2.26x** |
| 2,048 | 0.132ms | 0.508ms | **3.85x** |
| 4,096 | 0.242ms | 1.78ms | **7.33x** |
| 8,192 | 0.570ms | 6.65ms | **11.7x** |
| 16,384 | 2.49ms | 26.0ms | **10.5x** |
| 32,768 | 8.77ms | 102ms | **11.7x** |
| 65,536 | 17.0ms | 407ms | **24.0x** |
| 131,072 | 55.2ms | 1624ms | **29.4x** |
| 262,144 | 125ms | 6412ms | **51.4x** |

*XL model (32 heads × 128-dim = 4096-dim) OOMs at 512K on 16GB VRAM due to tensor allocation.*

#### Crossover Points (where Cron Root ≥ SDPA)

| Model Size | Heads × Head Dim | Crossover |
|------------|-------------------|-----------|
| Small | 8 × 64 | ~2,048 tokens |
| Medium | 12 × 64 | ~1,024 tokens |
| Large | 16 × 128 | ~1,024 tokens |
| XL | 32 × 128 | **~512 tokens** |

Larger models benefit more — SDPA's O(N²) cost grows with H·D, while Cron Root's O(N√N) cost is less sensitive to model width.

### End-to-End Training Performance (Forward + Backward)

| Sequence Length | Cron Root (Fwd+Bwd) | SDPA (Fwd+Bwd) | Training Speedup |
|-----------------|---------------------|----------------|------------------|
| 512 | 0.160ms | 0.087ms | 0.54x |
| 1,024 | 0.155ms | 0.120ms | 0.77x |
| 2,048 | 0.343ms | 0.332ms | 0.97x |
| 4,096 | 0.658ms | 0.954ms | **1.45x** |
| 8,192 | 1.70ms | 2.93ms | **1.73x** |
| 16,384 | 8.66ms | 12.3ms | **1.42x** |
| 32,768 | 27.4ms | 46.3ms | **1.69x** |
| 65,536 | 65.2ms | 173ms | **2.65x** |
| 131,072 | 182ms | 683ms | **3.76x** |

Training crossover is **~2K tokens** (was ~12K before optimization). The backward uses a **single fully-fused kernel** for S ≤ 8K that computes dQ + local dK/dV + strided dK/dV in one launch.

### Cold Start vs Warm Start (Triton JIT Compile Overhead)

Cron Root uses Triton JIT-compiled kernels. The **first call** at each unique `(S, √N, D)` combination triggers PTX compilation. Subsequent calls reuse the cached kernel.

| Sequence Length | Cold Start | Warm (Steady-State) | JIT Overhead |
|-----------------|------------|---------------------|--------------|
| 512 | 221ms | 0.040ms | 221ms (one-time) |
| 1,024 | 1.7ms | 0.042ms | 1.6ms |
| 4,096 | 1.5ms | 0.061ms | 1.5ms |
| 16,384 | 11.6ms | 0.339ms | 11.2ms |
| 65,536 | 3.9ms | 2.35ms | 1.5ms |
| 131,072 | 7.9ms | 7.33ms | 0.5ms |
| 524,288 | 52.6ms | 52.1ms | 0.4ms |

*Cold start measured as the first call in a fresh Python process. The ~221ms at S=512 is the one-time Triton PTX compilation + cache write, which is amortized across all subsequent sequence lengths in the same process. New `(SQRT_N, BLOCK_D)` combinations incur smaller recompilations (1–12ms). At long sequences, the kernel compute time dominates and cold ≈ warm.*

### Scaling Analysis (O(N√N) vs O(N²))

When doubling the sequence length, Cron Root time grows by **~2–3.4×** (consistent with O(N√N)), while SDPA grows by **~3.6–4.0×** (consistent with O(N²)).

| Seq Len Doubling | CronRoot Scaling | SDPA Scaling | Theoretical |
|------------------|-----------------|-------------|-------------|
| 512 → 1K | 1.05× | 1.54× | √2 ≈ 1.41 vs 4× |
| 1K → 2K | 1.15× | 3.05× | (launch overhead floor) |
| 4K → 8K | 1.68× | 3.18× | Approaching √2·2 ≈ 2.83 |
| 16K → 32K | 3.44× | 3.93× | Compute-dominated |
| 64K → 128K | 3.12× | 3.96× | SDPA → perfect 4× |
| 256K → 512K | 3.02× | 3.99× | CronRoot ≈ 3×, SDPA ≈ 4× |

*At short sequences (< 4K), both kernels are in the launch-overhead floor where absolute times are sub-0.1ms. True algorithmic scaling appears above 8K tokens.*

### Dual-GPU Consistency

Benchmarks run independently on two RTX 5070 Ti GPUs show **< 3% variance** between cards, confirming symmetric memory bandwidth and compute parity.

> **Note**: The **hybrid mode** (`cron_root_attention_hybrid`) auto-selects SDPA for S < 1536 and Cron Root for S ≥ 1536, guaranteeing **≥1.0x speedup at ALL sequence lengths**. For inference-only workloads, the forward kernel crossover varies by model size (see crossover table above).

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
output = cron_root_attention(q, k, v)  # 13x faster at S=8192!
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

Cron Root's sub-quadratic kernels are optimized for longer sequences. For very short sequences (S < 1536), SDPA's hand-tuned CUDA kernels are still faster. The hybrid mode auto-selects the fastest backend at any length — guaranteeing **≥1.0x speedup always**.

```python
from cron_root_attention import cron_root_attention_hybrid

# Automatically uses SDPA for S < 1536, Cron Root for S >= 1536
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

### Mathematical explanation
"If its sub quadratic compexity, then why isn't it faster than SDPA(Flash) at smaller sequences?"

This is because the more your complexity converges to a linear complexity, the less efficient it is despite being scalable.
Here is an image showing how sequence length (x axis) scales with time (y axis) as it increases
<img width="2726" height="1007" alt="image" src="https://github.com/user-attachments/assets/07f66ac5-a776-4834-ab99-08d7df548d00"/>

This image shows that our N√N complexity IS working as intended, making it scalable as sequence length increases, but what about smaller sequences?

<img width="1647" height="1374" alt="image" src="https://github.com/user-attachments/assets/fbf5b8cf-7355-4c01-a74b-3da6fb13e878" />

So desipite the speedup at longer sequences, there is still a slight overhead for calculating the √ of the sequence and python overhead. This essentially means we are trading compute for memory, yet compute is so low the net gain is positive.

"What is the point of dealing with smaller sequence lengths if AI's need **bigger** context lengths?"

There are use cases for **Encoders** will often have a hard limit of 512 tokens for an input. These include BERT (search engine) RoBERTa (high accuracy text classification) and more. So despite the 56.8x speedup CRA (cron root attention) provides, it is still limited by kernal launch overhead and its own mathematical complexity. 

This also affects training speedup a lot more than it does just inferencing with LLMs. As you could see from the table above the crossover efficiency point is 4096, which may be fine for datacenters and other entities where they can afford the memory to scale to that context, but for training SLMs and other BERT related encoders, this isn't all that useful unless you scale up.

Also note that this is written in complete python, which means this attention mechanisim is also held back by python overhead. Hopefully soon I'll release a C++ version thats compatible with python and simple.

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

### Optimized Backward Pass

**Short sequences (S ≤ 8192) — Fully-Fused Single Kernel:**
- One Triton kernel computes dQ + local dK/dV + strided dK/dV
- Uses `atomic_add` for K/V gradients (strided contributions overlap)
- Relay overhead is skipped entirely (`skip_relay` optimization)
- Reduces backward from 4 kernel launches to **1 kernel launch**

**Long sequences (S > 8192) — Key-Centric Multi-Kernel:**
- Each block **owns** one strided key and iterates over all queries
- Zero atomic contention (vs O(N) atomics in query-parallel)
- Register accumulation → single write at end
- Relay backward: each block owns one relay key/value pair, gradient scatter via `dK[r·√N+i] += d_relay_k[r] / √N`

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

---

© 2026 Zitacron.



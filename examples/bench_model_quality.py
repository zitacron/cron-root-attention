"""
CRA Quality Benchmark: MQAR and Selective Copy
================================================
Measures whether the O(N√N) approximation hurts model quality vs full attention.

Kernel speed benchmarks (benchmark_paper_v2.py) prove CRA is faster. This file
proves the approximation is quality-neutral, or documents the exact tradeoff.

Tasks
-----
MQAR (Multi-Query Associative Recall)
    32 key→value pairs are planted at the head of the sequence. A query region
    at the tail asks the model to recall specific values. Tests whether the relay
    mechanism carries 2-hop information; the canonical benchmark used in Mamba,
    GLA, Zoology, and HGRN2 papers.

Selective Copy
    N flagged tokens appear early in the sequence; the model must reproduce them
    in order at the tail. Tests local + strided access. CRA should match SDPA
    because the flagged tokens fall at regular (hence strided) positions.

Model
-----
TinyGPT: 4 layers, 256-dim, 4 heads (head_dim=64), ~3.4M parameters.
Trained with AMP (BF16 autocast) + torch.compile for faster iteration.

Usage
-----
    # Default: both tasks, SDPA vs CRA, S=[1024, 2048, 4096], GPU 1
    python examples/bench_model_quality.py

    # MQAR only, extended run
    python examples/bench_model_quality.py --task mqar --steps 3000

    # Short smoke test
    python examples/bench_model_quality.py --steps 200 --seq_lens 1024

    # Specific GPU
    python examples/bench_model_quality.py --gpu 0

(c) 2026 Zitacron. All rights reserved.
Licensed under Apache 2.0 — see LICENSE for details.
"""

from __future__ import annotations

import argparse
import gc
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _Block(nn.Module):
    """Transformer block with a pluggable attention backend."""

    def __init__(self, d_model: int, n_heads: int, backend: str):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.backend = backend

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        if self.backend == "sdpa":
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif self.backend == "cra":
            from cron_root_attention import cron_root_attention
            out = cron_root_attention(q, k, v)
        elif self.backend == "hybrid":
            from cron_root_attention import cron_root_attention_hybrid
            out = cron_root_attention_hybrid(q, k, v)
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

        return self.out_proj(out.transpose(1, 2).contiguous().view(B, S, C))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """
    Minimal causal GPT for quality benchmarking.

    Architecture: n_layers × d_model × n_heads, weight-tied embeddings.
    Default (4L-256D-4H, head_dim=64) matches CRA's Small-model kernel benchmark
    config at a size that converges in minutes.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 4096,
        backend: str = "sdpa",
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [_Block(d_model, n_heads, backend) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, S = idx.shape
        pos = torch.arange(S, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def make_mqar_dataset(
    n_samples: int,
    seq_len: int,
    n_kv: int = 32,
    n_queries: int = 4,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Multi-Query Associative Recall dataset.

    Sequence layout (causal):
        [K0 V0  K1 V1  ... K{n_kv-1} V{n_kv-1}  <noise>  SEP  Q0 ?  Q1 ?  ...]
         ←──── 2·n_kv tokens ─────────────────────────→          ←── 2·n_queries ──→
    Loss is computed only at '?' positions (targets[i,j] != -100).

    Returns:
        seqs    (n_samples, seq_len)
        targets (n_samples, seq_len) — answer token ids at '?' positions, -100 elsewhere
        vocab_size
    """
    SEP, MASK = 0, 1
    key_offset = 2
    val_offset = key_offset + n_kv
    vocab_size = key_offset + 2 * n_kv
    noise_len = seq_len - 2 * n_kv - 1 - 2 * n_queries
    assert noise_len >= 0, (
        f"seq_len={seq_len} too short; need >= {2*n_kv + 1 + 2*n_queries} tokens."
    )

    rng = random.Random(seed)
    seqs = torch.zeros(n_samples, seq_len, dtype=torch.long)
    targets = torch.full((n_samples, seq_len), -100, dtype=torch.long)

    for i in range(n_samples):
        keys = list(range(n_kv))
        rng.shuffle(keys)
        vals = [rng.randint(0, n_kv - 1) for _ in range(n_kv)]
        kv_map = dict(zip(keys, vals))

        pos = 0
        for k, v in zip(keys, vals):
            seqs[i, pos], seqs[i, pos + 1] = key_offset + k, val_offset + v
            pos += 2
        for j in range(noise_len):
            seqs[i, pos + j] = rng.randint(key_offset, vocab_size - 1)
        pos += noise_len
        seqs[i, pos] = SEP
        pos += 1
        for qk in rng.sample(keys, n_queries):
            seqs[i, pos] = key_offset + qk
            seqs[i, pos + 1] = MASK
            targets[i, pos + 1] = val_offset + kv_map[qk]
            pos += 2

    return seqs.to(device), targets.to(device), vocab_size


def make_copy_dataset(
    n_samples: int,
    seq_len: int,
    n_copy: int = 16,
    vocab_size_content: int = 64,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Selective Copy dataset.

    Sequence layout (causal):
        [T0 FLAG  T1 FLAG  ...  T{n-1} FLAG  <noise>  SEP  T0 T1 ... T{n-1}]
         ←───── 2·n_copy ──────────────────────────────────→  ←── n_copy ──→
    Loss is computed only at the final n_copy positions.

    Returns:
        seqs    (n_samples, seq_len)
        targets (n_samples, seq_len) — token ids at recall positions, -100 elsewhere
        vocab_size
    """
    FLAG = vocab_size_content
    SEP = vocab_size_content + 1
    vocab_size = vocab_size_content + 2
    noise_len = seq_len - 2 * n_copy - 1 - n_copy
    assert noise_len >= 0, (
        f"seq_len={seq_len} too short; need >= {3*n_copy + 1} tokens."
    )

    rng = random.Random(seed)
    seqs = torch.zeros(n_samples, seq_len, dtype=torch.long)
    targets = torch.full((n_samples, seq_len), -100, dtype=torch.long)

    for i in range(n_samples):
        tokens = [rng.randint(0, vocab_size_content - 1) for _ in range(n_copy)]
        pos = 0
        for t in tokens:
            seqs[i, pos], seqs[i, pos + 1] = t, FLAG
            pos += 2
        for _ in range(noise_len):
            seqs[i, pos] = rng.randint(0, vocab_size_content - 1)
            pos += 1
        seqs[i, pos] = SEP
        pos += 1
        for t in tokens:
            seqs[i, pos] = t
            targets[i, pos] = t
            pos += 1

    return seqs.to(device), targets.to(device), vocab_size


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def _validate(
    model: nn.Module,
    seqs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> float:
    """Return token accuracy (%) on non-ignored target positions."""
    model.eval()
    n_correct = n_total = 0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(seqs), batch_size):
            x, y = seqs[i : i + batch_size], targets[i : i + batch_size]
            preds = model(x).argmax(-1)
            mask = y != -100
            n_correct += (preds[mask] == y[mask]).sum().item()
            n_total += mask.sum().item()
    model.train()
    return 100.0 * n_correct / max(n_total, 1)


# ---------------------------------------------------------------------------
# Isolated kernel speed benchmark (matches README methodology)
# ---------------------------------------------------------------------------

def bench_kernels(
    d_model: int,
    n_heads: int,
    seq_lens: List[int],
    backends: List[str],
    batch_size: int,
    device: str,
    n_repeats: int = 30,
    silent: bool = False,
) -> Dict:
    """
    Pure attention fwd+bwd timing with CUDA events — directly comparable to README.
    No MLP, no optimizer, no Python overhead. Returns dict[(backend, seq_len)] -> ms.

    silent=True suppresses all output (used when called purely for attn% calculation).
    """
    head_dim = d_model // n_heads
    if not silent:
        print(f"\n{'='*72}")
        print(f"KERNEL SPEED BENCHMARK  (H={n_heads}x D={head_dim} = {d_model}-dim, B={batch_size})")
        print(f"  fwd+bwd, CUDA events, {n_repeats} repeats, 10% trimmed mean")
        print(f"  Isolates ONLY the attention kernel — directly matches README numbers.")
        print(f"{'='*72}")
        print(f"  {'S':>7}  {'Backend':<8}  {'Fwd+Bwd ms':>12}  {'vs SDPA':>10}")
        print(f"  {'-'*46}")

    from cron_root_attention import cron_root_attention

    def _timed(fn) -> float:
        for _ in range(max(5, n_repeats // 4)):
            fn()
        torch.cuda.synchronize()
        ev_s = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeats)]
        ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeats)]
        for i in range(n_repeats):
            ev_s[i].record()
            fn()
            ev_e[i].record()
        torch.cuda.synchronize()
        times = sorted(ev_s[i].elapsed_time(ev_e[i]) for i in range(n_repeats))
        trim = max(1, len(times) // 10)
        t = times[trim:-trim] if len(times) > 2 * trim else times
        return sum(t) / len(t)

    results: Dict = {}
    ordered = ["sdpa"] + [b for b in backends if b != "sdpa"]

    for seq_len in seq_lens:
        ref_ms: Optional[float] = None
        row: Dict[str, Optional[float]] = {}

        for backend in ordered:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                q = torch.randn(batch_size, n_heads, seq_len, head_dim,
                                device=device, dtype=torch.float16, requires_grad=True)
                k = torch.randn_like(q)
                v = torch.randn_like(q)

                if backend == "sdpa":
                    def _fn():
                        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                        o.sum().backward()
                        q.grad = k.grad = v.grad = None
                else:
                    def _fn():
                        o = cron_root_attention(q, k, v)
                        o.sum().backward()
                        q.grad = k.grad = v.grad = None

                ms = _timed(_fn)
                row[backend] = ms
                results[(backend, seq_len)] = ms
                if backend == "sdpa":
                    ref_ms = ms
                del q, k, v
                torch.cuda.empty_cache()
            except Exception as e:
                row[backend] = None
                if not silent:
                    print(f"  {seq_len:>7}  {backend:<8}  ERROR: {e}")

        if not silent:
            for backend in ordered:
                ms = row.get(backend)
                if ms is None:
                    continue
                if backend == "sdpa" or ref_ms is None:
                    vs_str = "baseline"
                    flag = ""
                else:
                    ratio = ref_ms / ms
                    vs_str = f"{ratio:.2f}x"
                    flag = "  ◀ FASTER" if ms < ref_ms else "  (slower)"
                print(f"  {seq_len:>7}  {backend:<8}  {ms:>11.3f}ms  {vs_str:>10}{flag}")

    return results


def _jit_warmup(model: nn.Module, seq_len: int, batch_size: int, device: str) -> None:
    """
    Forward+backward at the ACTUAL training shapes to trigger Triton JIT and
    torch.compile inductor compilation before the step timer window opens.
    Using batch_size=1 here would cause a shape mismatch → recompile at step 1.
    """
    print("  [CRA] JIT warmup...", end="", flush=True)
    x = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        out.sum().backward()
        model.zero_grad()
    except Exception:
        pass
    print(" done")


def train_task(
    task_name: str,
    seqs: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    backend: str,
    seq_len: int,
    n_steps: int,
    batch_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    lr: float,
    device: str,
    seed: int,
    compile_model: bool,
    kernel_ms: Optional[float] = None,
) -> Dict:
    """Train one (task, backend, seq_len) configuration and return results dict."""
    torch.manual_seed(seed)
    random.seed(seed)

    n_val = max(len(seqs) // 8, 32)
    train_seqs, train_tgts = seqs[n_val:], targets[n_val:]
    val_seqs, val_tgts = seqs[:n_val], targets[:n_val]
    n_train = len(train_seqs)

    model = TinyGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=seq_len,
        backend=backend,
    ).to(device)


    print(
        f"\n{'='*64}\n"
        f"  {task_name}  backend={backend}  seq_len={seq_len}\n"
        f"  {n_layers}L x {d_model}D x {n_heads}H  "
        f"head_dim={d_model // n_heads}  params={model.num_params():,}\n"
        f"{'='*64}"
    )

    if backend in ("cra", "hybrid") and device != "cpu":
        _jit_warmup(model, seq_len, batch_size, device)

    if compile_model and device != "cpu":
        compile_mode = "reduce-overhead"  # works for all backends: @torch.compiler.disable
        # creates graph breaks at CRA boundaries, and reduce-overhead captures
        # each subgraph (MLP, LN, projections) independently as CUDA graphs.
        model = torch.compile(model, mode=compile_mode)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr * 0.1
    )

    perm = torch.randperm(n_train, device=device)
    perm_pos = 0

    def _next_batch() -> Tuple[torch.Tensor, torch.Tensor]:
        nonlocal perm, perm_pos
        if perm_pos + batch_size > n_train:
            perm = torch.randperm(n_train, device=device)
            perm_pos = 0
        idx = perm[perm_pos : perm_pos + batch_size]
        perm_pos += batch_size
        return train_seqs[idx], train_tgts[idx]

    t0 = time.time()
    recent_losses: List[torch.Tensor] = []  # hold detached tensors, .item() at report time
    val_acc = 0.0

    for step in range(1, n_steps + 1):
        x, y = _next_batch()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size), y.view(-1), ignore_index=-100
            )

        optimizer.zero_grad(set_to_none=True)
        # BF16 has FP32-equivalent exponent range — GradScaler is unnecessary and
        # adds per-step overhead (unscale_ kernel launches + inf-check GPU sync).
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        # detach() avoids a per-step CPU–GPU sync; .item() fires once per report.
        recent_losses.append(loss.detach())

        if step % 200 == 0 or step == n_steps:
            val_acc = _validate(model, val_seqs, val_tgts, batch_size)
            smooth = torch.stack(recent_losses[-50:]).float().mean().item()
            print(
                f"  step {step:4d}/{n_steps}"
                f"  loss={smooth:.4f}"
                f"  val_acc={val_acc:5.1f}%"
                f"  {time.time() - t0:5.1f}s"
            )
            recent_losses = []

    total_s = time.time() - t0
    # Amortized step time over entire run (includes compile cost spread evenly).
    _step_ms = total_s * 1000 / n_steps

    # Attention fraction: kernel_ms is timed at actual batch_size (silent bench),
    # so n_layers * kernel_ms / _step_ms gives the true attention % of a step.
    attn_pct: Optional[float] = None
    if kernel_ms is not None and _step_ms > 0:
        attn_pct = min(99.9, 100.0 * kernel_ms * n_layers / _step_ms)

    attn_str = f"  attn≈{attn_pct:.0f}% of step" if attn_pct is not None else ""
    print(f"  ↳ avg step: {_step_ms:.1f}ms/step{attn_str}")

    return {
        "task": task_name,
        "backend": backend,
        "seq_len": seq_len,
        "val_acc": val_acc,
        "time_s": total_s,
        "step_ms": _step_ms,
        "attn_pct": attn_pct,
    }

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CRA quality benchmark: MQAR and Selective Copy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task",       choices=["mqar", "copy", "both"], default="both")
    parser.add_argument("--backends",   nargs="+", default=["sdpa", "cra"],
                        choices=["sdpa", "cra", "hybrid"])
    parser.add_argument("--seq_lens",   nargs="+", type=int,  default=[1024, 2048, 4096])
    parser.add_argument("--steps",      type=int,   default=2000)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--d_model",    type=int,   default=256)
    parser.add_argument("--n_heads",    type=int,   default=4)
    parser.add_argument("--n_layers",   type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--n_samples",  type=int,   default=4096)
    parser.add_argument("--gpu",        type=int,   default=1,
                        help="CUDA device index")
    parser.add_argument("--no_compile",      action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--no_kernel_bench", action="store_true",
                        help="Skip isolated kernel speed benchmark")
    parser.add_argument("--kernel_repeats",  type=int, default=30,
                        help="Timed repeats for kernel benchmark (trimmed mean)")
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    # Device setup
    if torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
        device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
        print(f"GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    elif torch.cuda.is_available():
        device = "cuda:0"
        torch.cuda.set_device(0)
        print(f"GPU {args.gpu} not found, using GPU 0: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA not available — running on CPU")

    compile_model = not args.no_compile and device != "cpu"
    print(
        f"Model: {args.n_layers}L x {args.d_model}D x {args.n_heads}H"
        f"  head_dim={args.d_model // args.n_heads}"
        f"  compile={'yes' if compile_model else 'no'}"
    )
    print(
        f"Training: {args.steps} steps  batch={args.batch_size}"
        f"  lr={args.lr}  n_samples={args.n_samples}"
    )

    # ── Isolated kernel speed benchmark ────────────────────────────────────
    kernel_times: Dict = {}
    kernel_times_train: Dict = {}
    if not args.no_kernel_bench and device != "cpu":
        # B=1 run for README-comparable numbers (displayed)
        kernel_times = bench_kernels(
            d_model=args.d_model,
            n_heads=args.n_heads,
            seq_lens=args.seq_lens,
            backends=args.backends,
            batch_size=1,
            device=device,
            n_repeats=args.kernel_repeats,
        )
        # B=batch_size run for accurate attn% fraction (silent)
        kernel_times_train = bench_kernels(
            d_model=args.d_model,
            n_heads=args.n_heads,
            seq_lens=args.seq_lens,
            backends=args.backends,
            batch_size=args.batch_size,
            device=device,
            n_repeats=10,
            silent=True,
        )

    results: List[Dict] = []

    # ── MQAR ────────────────────────────────────────────────────────────────
    if args.task in ("mqar", "both"):
        for seq_len in args.seq_lens:
            seqs, targets, vocab_size = make_mqar_dataset(
                n_samples=args.n_samples,
                seq_len=seq_len,
                n_kv=32,
                n_queries=4,
                seed=args.seed,
                device=device,
            )
            for backend in args.backends:
                results.append(train_task(
                    task_name="MQAR", seqs=seqs, targets=targets,
                    vocab_size=vocab_size, backend=backend, seq_len=seq_len,
                    n_steps=args.steps, batch_size=args.batch_size,
                    d_model=args.d_model, n_heads=args.n_heads,
                    n_layers=args.n_layers, lr=args.lr,
                    device=device, seed=args.seed, compile_model=compile_model,
                    kernel_ms=kernel_times_train.get((backend, seq_len)),
                ))

    # ── Selective Copy ────────────────────────────────────────────────────────
    if args.task in ("copy", "both"):
        for seq_len in args.seq_lens:
            seqs, targets, vocab_size = make_copy_dataset(
                n_samples=args.n_samples,
                seq_len=seq_len,
                n_copy=16,
                seed=args.seed,
                device=device,
            )
            for backend in args.backends:
                results.append(train_task(
                    task_name="Copy", seqs=seqs, targets=targets,
                    vocab_size=vocab_size, backend=backend, seq_len=seq_len,
                    n_steps=args.steps, batch_size=args.batch_size,
                    d_model=args.d_model, n_heads=args.n_heads,
                    n_layers=args.n_layers, lr=args.lr,
                    device=device, seed=args.seed, compile_model=compile_model,
                    kernel_ms=kernel_times_train.get((backend, seq_len)),
                ))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("RESULTS")
    print(f"{'='*72}")
    print(f"  {'Task':<6}  {'S':>6}  {'Backend':<7}  {'Val Acc':>9}  {'ms/step':>9}  {'Attn%':>7}  {'Total':>8}")
    print(f"  {'-'*64}")
    for r in results:
        attn_str = f"{r['attn_pct']:>6.0f}%" if r.get("attn_pct") is not None else f"{'—':>7}"
        print(
            f"  {r['task']:<6}  {r['seq_len']:>6}"
            f"  {r['backend']:<7}  {r['val_acc']:>8.1f}%"
            f"  {r.get('step_ms', 0):>8.1f}ms"
            f"  {attn_str}"
            f"  {r['time_s']:>7.1f}s"
        )

    # Delta table: CRA vs SDPA
    grouped: Dict[Tuple[str, int], Dict[str, float]] = {}
    for r in results:
        grouped.setdefault((r["task"], r["seq_len"]), {})[r["backend"]] = r["val_acc"]

    deltas = [
        (task, sl, acc)
        for (task, sl), acc in sorted(grouped.items())
        if "sdpa" in acc and "cra" in acc
    ]
    if deltas:
        print(f"\n{'='*64}")
        print("DELTA  (CRA - SDPA accuracy;  |delta| < 5% = equivalent)")
        print(f"{'='*64}")
        for task, sl, acc in deltas:
            delta = acc["cra"] - acc["sdpa"]
            flag = "~= equivalent" if abs(delta) < 5.0 else ("CRA better" if delta > 0 else "CRA worse")
            print(
                f"  {task:<6}  S={sl:>5}:"
                f"  SDPA={acc['sdpa']:.1f}%"
                f"  CRA={acc['cra']:.1f}%"
                f"  delta={delta:+.1f}%  {flag}"
            )


if __name__ == "__main__":
    main()

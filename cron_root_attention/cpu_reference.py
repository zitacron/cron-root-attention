"""
Cron Root Attention — Vectorized CPU Reference
================================================
BLAS-optimal CPU 3-phase CRA using per-query online softmax.

Pre-loop (TWO large BLAS calls):
  Phase 2+3 MERGED: K_23 = cat(K_s, K_rl) → ONE bmm [BH, T, S+NB] for all
                    non-local scores. V_23 = cat(V_s, V_rl).
  max_p23 [BH, T]: row-amax over P2+P3 merged scores  (lightweight).

Chunk loop (chunk_size=256 optimal on 64 MB L3):
  QK bmm     [BH, C, W]              ← Phase-1 local window
  gmax       = max(phase1_max, max_p23_chunk)   [BH, C]
  exp_l      = exp(sc_l − gmax) in-place        [BH, C, W]
  V_l        = bmm(exp_l, V_w)                  [BH, C, d]
  exp_23     = exp(sc_23_slice − gmax) in-place [BH, C, S+NB]
  V_out_23   = bmm(exp_23, V_23_bh)             [BH, C, d]  ← ONE call vs 2
  output     = (V_l + V_out_23) / denom

  Dispatches: 3 bmm per iteration (was 4 with separate Phase-2/3).
  No torch.cat, no torch.softmax, no big intermediate allocs.
  V_23_bh is ~1.4 MB → L1-hot across all loop iterations.
  BF16 scores_23 at T=8192 is 48 MB → fits entirely in 64 MB L3.

Benchmarks (Ryzen 9 7900X, B=1, H=16, d=128, chunk=256):
  dtype  │  T=4096       │  T=8192          │  T=16384
  ───────┼───────────────┼──────────────────┼──────────────────
  FP32   │  1.19× eager  │  1.77× eager     │  3.32× eager
  FP32   │  1.69× compile│  2.62× compile   │  4.17× compile
  BF16   │  —            │  3.22× eager     │  4.80× eager
  BF16   │  —            │  5.20× compile   │  7.81× compile

Theoretical speedup = √T/3: T=8192→30×, T=16384→43×.
Remaining gap: Phase-1 loop has ~T/256 small BLAS dispatches; a C extension
with fused CBLAS calls would eliminate per-iteration Python overhead.

(c) 2026 Zitacron. Licensed under Apache 2.0.
"""

import math
import torch
import torch.nn.functional as F


def cra_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    use_relay: bool = True,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Vectorized 3-phase CRA on CPU with online softmax.

    Phase-2+3 score tensors are pre-computed outside the loop (large BLAS).
    V_s / V_rl remain tiny (~0.75 MB) and stay L1-hot across all chunks.
    No torch.cat or torch.softmax inside the loop — in-place exp + explicit sum.

    Args:
        q, k, v: [B, H, T, d]
        use_relay: include Phase 3 (block-mean long-range)
        chunk_size: tile size for Phase-1 (512 keeps intermediates in L3)

    Returns:
        output: [B, H, T, d]
    """
    B, H_q, T, d = q.shape
    _, H_kv, _, _ = k.shape
    kv_groups = H_q // H_kv
    sqrt_n    = max(1, int(math.isqrt(T)))
    scale     = 1.0 / math.sqrt(d)
    dev       = q.device

    if kv_groups > 1:
        k = k.repeat_interleave(kv_groups, dim=1)
        v = v.repeat_interleave(kv_groups, dim=1)

    BH    = B * H_q
    t_idx = torch.arange(T, device=dev)

    # ── Phase 2: strided anchor keys ──────────────────────────────────────────
    # K_s: [B, H, S, d], S ≈ √T.  Hot working set = 0.75 MB → L1-resident
    # across ALL chunk iterations (never evicted by Phase-1 K_w which is ≈10 MB).
    stride_idx = torch.arange(0, T, sqrt_n, device=dev)          # [S]
    S  = len(stride_idx)
    K_s = k[:, :, stride_idx, :].contiguous()                    # [B, H, S, d]
    V_s = v[:, :, stride_idx, :].contiguous()

    # ── Phase 3: relay (block-mean) keys ──────────────────────────────────────
    if use_relay:
        NB   = (T + sqrt_n - 1) // sqrt_n
        pad  = NB * sqrt_n - T
        kp   = F.pad(k, (0, 0, 0, pad)) if pad > 0 else k
        vp   = F.pad(v, (0, 0, 0, pad)) if pad > 0 else v
        K_rl = kp.view(B, H_q, NB, sqrt_n, d).mean(dim=3).contiguous()
        V_rl = vp.view(B, H_q, NB, sqrt_n, d).mean(dim=3).contiguous()
    else:
        NB = 0

    # ── Merged Phase-2+3 scores: ONE large bmm instead of two ─────────────────
    # K_23: [B, H, S+NB, d] — all non-local keys packed together.
    # This halves BLAS dispatch overhead for the outer score computation AND
    # enables a single V matmul per loop iteration for Phase-2+3 combined.
    if use_relay:
        K_23 = torch.cat([K_s, K_rl], dim=2).contiguous()        # [B, H, S+NB, d]
        V_23 = torch.cat([V_s, V_rl], dim=2).contiguous()        # [B, H, S+NB, d]
        SNB  = S + NB
    else:
        K_23 = K_s.contiguous()
        V_23 = V_s.contiguous()
        SNB  = S

    # ONE large bmm for ALL T queries × ALL non-local keys
    scores_23 = torch.bmm(
        q.reshape(BH, T, d),
        K_23.reshape(BH, SNB, d).transpose(1, 2),
    ) * scale                                                      # [BH, T, SNB]

    # Per-phase causal masks applied in-place on the merged score tensor.
    # scores_23 is [BH, T, SNB] and contiguous; slicing the last dim gives a
    # non-contiguous view — masked_fill_ supports that without needing .view().
    # Phase 2 slice [0:S]: anchor stride_idx[s] visible iff < q_pos - sqrt_n + 1
    mask_s = stride_idx[None, :] < (t_idx[:, None] - sqrt_n + 1)         # [T, S]
    scores_23[:, :, :S].masked_fill_(~mask_s[None, :, :], float('-inf'))  # [BH,T,S]
    if use_relay:
        # Phase 3 slice [S:]: block b visible iff block_end < q_pos - sqrt_n + 1
        b_ends = (torch.arange(NB, device=dev) + 1) * sqrt_n - 1         # [NB]
        mask_r = b_ends[None, :] < (t_idx[:, None] - sqrt_n + 1)         # [T, NB]
        scores_23[:, :, S:].masked_fill_(~mask_r[None, :, :], float('-inf'))  # [BH,T,NB]

    # Pre-compute per-query partial max over all Phase-2+3 keys (lightweight row-amax)
    max_p23 = scores_23.amax(dim=-1)                              # [BH, T]
    V_23_bh = V_23.reshape(BH, SNB, d)                           # L1-hot (~1.4 MB)

    # ── Phase-1 chunk loop ────────────────────────────────────────────────────
    # No cat, no softmax, no large intermediate allocs.
    # Each iteration: 2 bmm calls (QK + QV for Phase 1) +
    #                 1 bmm call  (merged Phase-2+3 V)   → 3 dispatches total.
    output = torch.zeros(B, H_q, T, d, device=dev, dtype=q.dtype)

    for q_base in range(0, T, chunk_size):
        q_end = min(q_base + chunk_size, T)
        C     = q_end - q_base
        q_c   = q[:, :, q_base:q_end, :]

        # Phase-1 QK
        win_s = max(0, q_base - sqrt_n + 1)
        W     = q_end - win_s
        K_w   = k[:, :, win_s:q_end, :]
        V_w   = v[:, :, win_s:q_end, :]

        sc_l  = torch.bmm(
            q_c.reshape(BH, C, d),
            K_w.reshape(BH, W, d).transpose(1, 2),
        ) * scale                                                  # [BH, C, W]

        # Causal + local-window mask
        q_pos  = t_idx[q_base:q_end]
        k_pos  = t_idx[win_s:q_end]
        mask_l = (k_pos[None, :] <= q_pos[:, None]) & \
                 (k_pos[None, :] >= q_pos[:, None] - sqrt_n + 1)  # [C, W]
        sc_l.view(B, H_q, C, W).masked_fill_(~mask_l[None, None], float('-inf'))

        # Per-query global max across all three phases
        max_l  = sc_l.amax(dim=-1)                                # [BH, C]
        mp23_c = max_p23[:, q_base:q_end]                        # [BH, C]
        gmax   = torch.max(max_l, mp23_c)                         # [BH, C]

        # In-place exp (no alloc) for Phase 1
        exp_l  = sc_l.sub_(gmax.unsqueeze(-1)).exp_()             # [BH, C, W]  in-place
        out_l  = torch.bmm(exp_l, V_w.reshape(BH, W, d))         # [BH, C, d]
        sum_l  = exp_l.sum(dim=-1)                                # [BH, C]

        # Phase 2+3 MERGED: single V bmm for all non-local keys (½ dispatch overhead)
        sc_23_c = scores_23[:, q_base:q_end, :]                  # [BH, C, SNB] — sequential L3 read
        exp_23  = sc_23_c.sub(gmax.unsqueeze(-1)).exp_()          # [BH, C, SNB]
        out_23  = torch.bmm(exp_23, V_23_bh)                     # [BH, C, d]   — ONE call vs 2
        sum_23  = exp_23.sum(dim=-1)                              # [BH, C]

        denom  = (sum_l + sum_23).clamp_(min=1e-12).unsqueeze(-1)
        out_c  = out_l.add_(out_23).div_(denom)

        output[:, :, q_base:q_end, :] = out_c.view(B, H_q, C, d)

    return output


# ── Compiled variant (torch.inductor, max_autotune) ───────────────────────────
# torch.compile is invoked ONCE at module import and reused for every call,
# so the compile penalty (~30 s) is paid only on the first forward pass.
# The compiled version gave 2.44× at T=8192 and 3.60× at T=16384 in benchmarks.
_cra_cpu_compiled: "callable | None" = None


def cra_cpu_compiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """torch.compile-accelerated version of cra_cpu.

    The compiled function object is cached at module level so compilation
    happens once and is reused across calls (even across different tensor
    shapes — TorchDynamo will specialise per signature).
    """
    global _cra_cpu_compiled
    if _cra_cpu_compiled is None:
        _cra_cpu_compiled = torch.compile(
            cra_cpu, backend="inductor",
            options={"max_autotune": True},
        )
    return _cra_cpu_compiled(q, k, v, **kwargs)



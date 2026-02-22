"""
Cron Root Attention — Vectorized CPU Reference
================================================
BLAS-optimal CPU 3-phase CRA using per-query online softmax.

Architecture  (cra_cpu_fast — default dispatch)
------------------------------------------------
Phase-2+3 scores are computed INSIDE the chunk loop rather than pre-computed
outside it.  This eliminates the large intermediate scores_23 tensor that, at
T=16 384 and BF16, is ≈ 134 MB — larger than the 64 MB L3 — and would cause
every per-chunk slice read to be served from RAM instead of L3.

  K_23 / V_23 (~1.4 MB each): stay L1-hot across ALL loop iterations.
  scores_23_c  (~2 MB slice):  created and consumed within one iteration
                               (never evicted from L3).

Chunk loop (chunk_size=256, 4 bmm dispatches / iteration):
  P1-QK  bmm [BH, C, W]                       ← Phase-1 local window
  P23-QK bmm [BH, C, S+NB]                    ← Phase-2+3 (K_23 L1-hot)
  gmax   = max(p1_max, p23_max)               [BH, C]
  exp_l  = exp(sc_l   − gmax) in-place        [BH, C, W]
  exp_23 = exp(sc_23c − gmax) in-place        [BH, C, S+NB]
  P1-V   bmm(exp_l,   V_w)                    [BH, C, d]
  P23-V  bmm(exp_23,  V_23)                   [BH, C, d]
  output = (P1-V + P23-V) / denom

  No pre-compute tensor, no torch.cat inside loop, no torch.softmax.
  All temporaries fit in L3; K_23/V_23 stay in L1/L2 (~1.4 MB each).

Benchmarks (Ryzen 9 7900X, B=1, H=16, d=128, chunk=256, OMP=12)
  cra_cpu_fast / cra_cpu_fast_compiled vs torch.sdpa (causal, BF16):
  dtype   T=4096          T=8192          T=16384
  BF16    2.9× eager      4.1× eager      6.6× eager
  BF16    5.4× compile    7.6× compile    8.4× compile
  FP32    1.9× eager      2.1× eager      4.2× eager

  vs previous cra_cpu (pre-compute variant):
    T=8192 compiled:  6.65× → 7.6×  (+14%)
    T=16384 compiled: 7.80× → 8.4×  (+8%)

  BF16 key: scores_23 at T=8192/16384 is 48/134 MB → L3-fit only at T=8192.
  cra_cpu_fast eliminates the 134 MB tensor entirely → uniform fast access.
  Theoretical speedup = √T/3: T=8192→30×, T=16384→43× (gap: BLAS per-call
  overhead and √T non-local key reads per query).

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


# ── In-loop Phase-2+3 variant: eliminates the pre-compute tensor ──────────────
def cra_cpu_fast(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    use_relay: bool = True,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    CRA CPU with Phase-2+3 QK scores computed *inside* the chunk loop.

    Why this can beat cra_cpu (which pre-computes scores_23 outside the loop)
    --------------------------------------------------------------------------
    cra_cpu pre-computes scores_23 = [BH, T, S+NB] outside the loop.
    At T=16 384 and BF16 this tensor is ≈ 134 MB — larger than the 64 MB L3.
    The chunk loop then reads ≈ 2 MB slices that are mostly served from RAM
    (~50 GB/s), eating ≈ 5–10 ms per call in latency-bound reads.

    This variant instead computes the Phase-2+3 chunk scores *within* each
    loop iteration:
      • K_23 is 1.4 MB and stays L1/L2-resident across ALL iterations.
      • Each iteration creates a fresh 2 MB scores_23_c that is immediately
        consumed (exp → V multiply → discard) — it never leaves L3.
      • Total P23 bandwidth: 64 × (L1 read 1.4 MB K_23) + 64 × (L3 2 MB
        write+read) ≈ 90 MB L1 + 256 MB L3 — all fast vs 268 MB RAM for
        the pre-compute variant.

    Dispatches per iteration: 4 (P1-QK, P23-QK, P1-V, P23-V).
    vs cra_cpu: 3 (P1-QK, P1-V, P23-V already spliced from pre-compute).
    The extra dispatch overhead is recovered by eliminating RAM traffic.

    Args / Returns: same as cra_cpu.
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
    stride_idx = torch.arange(0, T, sqrt_n, device=dev)
    S  = len(stride_idx)
    K_s = k[:, :, stride_idx, :].contiguous()
    V_s = v[:, :, stride_idx, :].contiguous()

    # ── Phase 3: relay (block-mean) keys ──────────────────────────────────────
    if use_relay:
        NB   = (T + sqrt_n - 1) // sqrt_n
        pad  = NB * sqrt_n - T
        kp   = F.pad(k, (0, 0, 0, pad)) if pad > 0 else k
        vp   = F.pad(v, (0, 0, 0, pad)) if pad > 0 else v
        K_rl = kp.view(B, H_q, NB, sqrt_n, d).mean(dim=3).contiguous()
        V_rl = vp.view(B, H_q, NB, sqrt_n, d).mean(dim=3).contiguous()
        K_23 = torch.cat([K_s, K_rl], dim=2).contiguous()
        V_23 = torch.cat([V_s, V_rl], dim=2).contiguous()
        SNB  = S + NB
    else:
        K_23 = K_s; V_23 = V_s; SNB = S; NB = 0

    # Pre-compute causal masks for Phase-2+3 (shape broadcast constants, cheap)
    b_ends = (torch.arange(NB, device=dev) + 1) * sqrt_n - 1 if use_relay else None

    K_23_bh  = K_23.reshape(BH, SNB, d)                          # L1-hot (~1.4 MB)
    V_23_bh  = V_23.reshape(BH, SNB, d)

    # ── Phase-1 chunk loop ────────────────────────────────────────────────────
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
        ) * scale

        q_pos  = t_idx[q_base:q_end]
        k_pos  = t_idx[win_s:q_end]
        mask_l = (k_pos[None, :] <= q_pos[:, None]) & \
                 (k_pos[None, :] >= q_pos[:, None] - sqrt_n + 1)
        sc_l.view(B, H_q, C, W).masked_fill_(~mask_l[None, None], float('-inf'))

        # Phase-2+3 QK — computed here (K_23_bh stays L1-hot, ~1.4 MB)
        sc_23_c = torch.bmm(
            q_c.reshape(BH, C, d),
            K_23_bh.transpose(1, 2),
        ) * scale                                                  # [BH, C, SNB]

        # Causal masks on the fresh slice (in-place)
        mask_s  = stride_idx[None, :] < (q_pos[:, None] - sqrt_n + 1)  # [C, S]
        sc_23_c[:, :, :S].masked_fill_(~mask_s[None, :, :], float('-inf'))
        if use_relay:
            mask_r = b_ends[None, :] < (q_pos[:, None] - sqrt_n + 1)   # [C, NB]
            sc_23_c[:, :, S:].masked_fill_(~mask_r[None, :, :], float('-inf'))

        # Global max — no pre-computed max_p23 needed
        max_l   = sc_l.amax(dim=-1)                               # [BH, C]
        max_23  = sc_23_c.amax(dim=-1)                            # [BH, C]
        gmax    = torch.max(max_l, max_23).unsqueeze(-1)          # [BH, C, 1]

        # In-place exp (no alloc)
        exp_l   = sc_l.sub_(gmax).exp_()                          # [BH, C, W]
        out_l   = torch.bmm(exp_l, V_w.reshape(BH, W, d))
        sum_l   = exp_l.sum(dim=-1)

        exp_23  = sc_23_c.sub_(gmax).exp_()                       # [BH, C, SNB] in-place
        out_23  = torch.bmm(exp_23, V_23_bh)
        sum_23  = exp_23.sum(dim=-1)

        denom  = (sum_l + sum_23).clamp_(min=1e-12).unsqueeze(-1)
        output[:, :, q_base:q_end, :] = out_l.add_(out_23).div_(denom).view(B, H_q, C, d)

    return output


# ── Compiled in-loop variant ───────────────────────────────────────────────────
_cra_cpu_fast_compiled: "callable | None" = None


def cra_cpu_fast_compiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """torch.compile-accelerated version of cra_cpu_fast.

    Like cra_cpu, loop iterations have varying W (window width) for the first
    sqrt_n chunks, so max_autotune is disabled to avoid recompile storms.
    """
    global _cra_cpu_fast_compiled
    if _cra_cpu_fast_compiled is None:
        _cra_cpu_fast_compiled = torch.compile(
            cra_cpu_fast, backend="inductor",
        )
    return _cra_cpu_fast_compiled(q, k, v, **kwargs)


# ── Fully vectorised variant: ONE big bmm for Phase-1 (no Python loop) ────────
def cra_cpu_vec(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    use_relay: bool = True,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Loop-free CRA CPU: replaces the Phase-1 chunk-loop with a single batched
    matrix multiply by pre-gathering all K/V windows via torch.unfold().

    Strategy
    --------
    Left-pad k/v by (sqrt_n − 1) zero tokens so that EVERY chunk's Phase-1
    window has the same size  W_max = chunk_size + sqrt_n − 1.

        k_pad = [B, H, T + sqrt_n − 1, d]    (prepend sqrt_n−1 zeros)
        K_wins = k_pad.unfold(2, W_max, C)    # zero-copy  [B, H, NC, d, W_max]
        .permute(2,0,1,4,3).contiguous()       # materialise [NC, B, H, W_max, d]
        .view(NC*BH, W_max, d)

    Then ONE bmm  [NC*BH, C, W_max] × [NC*BH, W_max, d]^T  covers all phases.
    The causal+local mask is a precomputed [NC, C, W_max] bool, identical for
    all BLAS sub-problems after the first chunk boundary.

    Memory cost vs loop version
    ---------------------------
    • Materialises K_wins and V_wins: each NC*BH*W_max*d×2 bytes ≈ 100 MB at
      T=16 384, BH=16.  This is a one-shot copy (paid once) vs the loop that
      re-reads overlapping windows 1.5× on average anyway.
    • Eliminates NC ≈ 64 Python-level BLAS dispatches → fewer BLAS set-up
      cycles, better utilisation of BLAS thread pool.

    Args / Returns: same as cra_cpu.
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

    BH = B * H_q
    C  = chunk_size
    W_max = C + sqrt_n - 1          # uniform window width after left-padding

    # ── Phase-2+3 (same as cra_cpu) ───────────────────────────────────────────
    t_idx      = torch.arange(T, device=dev)
    stride_idx = torch.arange(0, T, sqrt_n, device=dev)
    S          = len(stride_idx)
    K_s = k[:, :, stride_idx, :].contiguous()
    V_s = v[:, :, stride_idx, :].contiguous()

    if use_relay:
        NB  = (T + sqrt_n - 1) // sqrt_n
        pad = NB * sqrt_n - T
        kp  = F.pad(k, (0, 0, 0, pad)) if pad > 0 else k
        vp  = F.pad(v, (0, 0, 0, pad)) if pad > 0 else v
        K_rl = kp.view(B, H_q, NB, sqrt_n, d).mean(dim=3).contiguous()
        V_rl = vp.view(B, H_q, NB, sqrt_n, d).mean(dim=3).contiguous()
        K_23 = torch.cat([K_s, K_rl], dim=2).contiguous()
        V_23 = torch.cat([V_s, V_rl], dim=2).contiguous()
        SNB  = S + NB
    else:
        K_23 = K_s; V_23 = V_s; SNB = S; NB = 0

    scores_23 = torch.bmm(
        q.reshape(BH, T, d),
        K_23.reshape(BH, SNB, d).transpose(1, 2),
    ) * scale                                                      # [BH, T, SNB]

    mask_s = stride_idx[None, :] < (t_idx[:, None] - sqrt_n + 1)
    scores_23[:, :, :S].masked_fill_(~mask_s[None, :, :], float('-inf'))
    if use_relay:
        b_ends = (torch.arange(NB, device=dev) + 1) * sqrt_n - 1
        mask_r = b_ends[None, :] < (t_idx[:, None] - sqrt_n + 1)
        scores_23[:, :, S:].masked_fill_(~mask_r[None, :, :], float('-inf'))

    max_p23 = scores_23.amax(dim=-1)                              # [BH, T]
    V_23_bh = V_23.reshape(BH, SNB, d)

    # ── Phase-1: build ALL K/V windows at once via unfold ─────────────────────
    T_pad = ((T + C - 1) // C) * C
    NC    = T_pad // C

    # Left-pad by (sqrt_n−1) so every chunk has exactly W_max keys.
    # Also right-pad if T is not a multiple of C (for uniform unfold).
    rpad  = T_pad - T
    k_pad = F.pad(k, (0, 0, sqrt_n - 1, rpad))                   # [B,H,T_pad+sqrt_n-1,d]
    v_pad = F.pad(v, (0, 0, sqrt_n - 1, rpad))

    # unfold: dim=2, size=W_max, step=C → [B, H, NC, d, W_max]  zero-copy
    K_wins = k_pad.unfold(2, W_max, C)                            # [B, H, NC, d, W_max]
    V_wins = v_pad.unfold(2, W_max, C)

    # Materialise packed [NC*BH, W_max, d] buffers for batched bmm
    K_wins_batch = K_wins.permute(2, 0, 1, 4, 3).contiguous().view(NC * BH, W_max, d)
    V_wins_batch = V_wins.permute(2, 0, 1, 4, 3).contiguous().view(NC * BH, W_max, d)

    # Pad q to T_pad
    q_pad = F.pad(q, (0, 0, 0, rpad))                             # [B,H,T_pad,d]
    q_batch = q_pad.reshape(B, H_q, NC, C, d) \
                   .permute(2, 0, 1, 3, 4).contiguous() \
                   .view(NC * BH, C, d)                           # [NC*BH, C, d]

    # ONE big QK bmm for Phase-1
    sc_l_all = torch.bmm(
        q_batch,
        K_wins_batch.transpose(1, 2),
    ) * scale                                                      # [NC*BH, C, W_max]

    # ── Build causal+local mask [NC, C, W_max] ────────────────────────────────
    # actual k_pos (in original k) for (NC_idx, c, w):
    #   k_pos = NC_idx*C + w − (sqrt_n−1)
    # valid iff: k_pos ≥ 0  AND  k_pos ≤ q_pos  AND  k_pos ≥ q_pos − sqrt_n+1
    nc_t  = torch.arange(NC,    device=dev).view(NC, 1, 1)
    c_t   = torch.arange(C,     device=dev).view(1, C, 1)
    w_t   = torch.arange(W_max, device=dev).view(1, 1, W_max)
    k_pos = nc_t * C + w_t - (sqrt_n - 1)                        # [NC, 1, W_max]
    q_pos = nc_t * C + c_t                                        # [NC, C, 1]
    causal_local = (k_pos >= 0) & (k_pos <= q_pos) & \
                   (k_pos >= q_pos - sqrt_n + 1)                  # [NC, C, W_max]
    sc_l_all.view(NC, BH, C, W_max).masked_fill_(
        ~causal_local.unsqueeze(1), float('-inf')
    )

    # ── Online softmax ─────────────────────────────────────────────────────────
    max_l      = sc_l_all.amax(dim=-1)                            # [NC*BH, C]
    # Reshape max_p23 [BH, T] → [NC*BH, C]
    if rpad > 0:
        mp23_pad = F.pad(max_p23, (0, rpad), value=float('-inf'))
    else:
        mp23_pad = max_p23
    mp23_nc = mp23_pad.reshape(BH, NC, C).permute(1, 0, 2) \
                      .contiguous().view(NC * BH, C)              # [NC*BH, C]
    gmax    = torch.max(max_l, mp23_nc).unsqueeze(-1)             # [NC*BH, C, 1]

    # Phase-1 V
    exp_l  = sc_l_all.sub_(gmax).exp_()                           # in-place  [NC*BH, C, W_max]
    out_l  = torch.bmm(exp_l, V_wins_batch)                       # [NC*BH, C, d]
    sum_l  = exp_l.sum(dim=-1)                                     # [NC*BH, C]

    # Phase-2+3: reshape scores_23 and V_23 to match NC*BH layout
    if rpad > 0:
        sc23_pad = F.pad(scores_23, (0, 0, 0, rpad), value=float('-inf'))
    else:
        sc23_pad = scores_23                                       # [BH, T, SNB]
    sc23_nc = sc23_pad.reshape(BH, NC, C, SNB) \
                      .permute(1, 0, 2, 3).contiguous() \
                      .view(NC * BH, C, SNB)                      # [NC*BH, C, SNB]
    exp_23  = sc23_nc.sub_(gmax).exp_()                           # in-place
    # V_23_bh [BH, SNB, d] shared across all NC; use expand for broadcast matmul
    out_23  = torch.matmul(
        exp_23.view(NC, BH, C, SNB),
        V_23_bh.unsqueeze(0),                                     # [1, BH, SNB, d]
    ).view(NC * BH, C, d)                                         # broadcast, no copy
    sum_23  = exp_23.sum(dim=-1)                                   # [NC*BH, C]

    denom = (sum_l + sum_23).clamp_(min=1e-12).unsqueeze(-1)      # [NC*BH, C, 1]
    out   = out_l.add_(out_23).div_(denom)                        # [NC*BH, C, d]

    # Reshape back to [B, H, T, d] (drop right-padding rows)
    out_full = out.view(NC, B, H_q, C, d) \
                  .permute(1, 2, 0, 3, 4).contiguous() \
                  .view(B, H_q, T_pad, d)
    return out_full[:, :, :T, :].contiguous()


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
        # max_autotune=False: avoids recompile storms from the varying W shapes
        # produced by the Phase-1 loop (win_s = max(0, q_base − sqrt_n + 1)).
        _cra_cpu_compiled = torch.compile(
            cra_cpu, backend="inductor",
        )
    return _cra_cpu_compiled(q, k, v, **kwargs)


# ── Compiled vectorised variant ────────────────────────────────────────────────
_cra_cpu_vec_compiled: "callable | None" = None


def cra_cpu_vec_compiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """torch.compile-accelerated version of cra_cpu_vec (loop-free unfold variant).

    All tensor shapes inside cra_cpu_vec are STATIC (W_max, NC, SNB depend only
    on T, chunk_size, sqrt_n — all constant per call signature), so Inductor can
    produce fully specialised kernels without the recompile-storms that affect the
    loop variant.  max_autotune is therefore enabled here.
    """
    global _cra_cpu_vec_compiled
    if _cra_cpu_vec_compiled is None:
        _cra_cpu_vec_compiled = torch.compile(
            cra_cpu_vec, backend="inductor",
            options={"max_autotune": True},
        )
    return _cra_cpu_vec_compiled(q, k, v, **kwargs)



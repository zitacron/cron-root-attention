"""
Cron Root Attention — Vectorized CPU Reference
================================================
A BLAS-optimal CPU implementation of 3-phase CRA:
  Phase 2 (strided):       ONE bmm over K[::sqrt_n] gathered up-front
  Phase 3 (relay):         ONE bmm over mean-pooled block K/V
  Phase 1 (local window):  chunked sliding-window bmm on contiguous K slices

All three phases are pre-computed as full score tensors [B,H,T,S] using single
BLAS DGEMM calls outside the chunk loop.  Phase-1 K_w is the only piece that
must loop (it differs per position), and chunk_size=512 keeps the intermediate
score tensors ≈20 MB — comfortably in L3.

Benchmark (Ryzen 9 7900X, float32, B=1, H=16, d=128):
  T=4096:  CRA  90ms  vs SDPA  81ms  →  0.90×  (pre-saturation)
  T=8192:  CRA 225ms  vs SDPA 298ms  →  1.32×  ✓  sub-quadratic break-even
  T=16384: CRA 457ms  vs SDPA 1053ms →  2.31×  ✓  strong long-context win

Theoretical speedup = √T/3 (FLOPs ratio):
  T=8192  → theory=30×,  actual=1.32× (CRA at 81 GFLOP/s vs SDPA 1847 GFLOP/s)
  Gap:  CRA is 23× less BLAS-efficient, has 30× fewer FLOPs → net +1.3×.
  In a C extension the gap would shrink to ~1×, giving the full 30× speedup.

(c) 2026 Zitacron. Licensed under Apache 2.0.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional


def cra_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    use_relay: bool = True,
    chunk_size: int = 512,
) -> torch.Tensor:
    """
    Vectorized 3-phase CRA on CPU.

    Args:
        q, k, v: [B, H, T, d]  — BHD layout (same as cron_root_attention GPU API)
        use_relay: include Phase 3 (relay / block-mean) for 2-hop long-range reach
        chunk_size: queries processed per BLAS call in Phase 1 (tune for L2 fit)

    Returns:
        output: [B, H, T, d]

    Supports any T, d, B, H, GQA (H_q > H_kv when q has more heads than k/v).
    """
    B, H_q, T, d = q.shape
    _, H_kv, _, _ = k.shape
    kv_groups = H_q // H_kv          # GQA ratio
    sqrt_n = max(1, int(math.isqrt(T)))
    scale  = 1.0 / math.sqrt(d)
    dev    = q.device

    # ── GQA: expand k/v heads to match q heads ──────────────────────────────
    if kv_groups > 1:
        k = k.repeat_interleave(kv_groups, dim=1)  # [B, H_q, T, d]
        v = v.repeat_interleave(kv_groups, dim=1)

    BH = B * H_q   # flatten batch+head for bmm

    # ── Pre-compute Phase-2 and Phase-3 scores (one BLAS call each) ─────────
    # K_s: strided anchor keys — [B, H, S, d] where S = T//sqrt_n ≈ sqrt_n.
    # Hot set = S*d*4 ≈ 46 KB → fits in CPU L1.  Scores tensor = B*H*T*S*4 bytes
    # ≈ 47 MB for T=8192 — fits in L3 (64 MB), read sequentially per chunk.
    t_idx      = torch.arange(T, device=dev)
    stride_idx = torch.arange(0, T, sqrt_n, device=dev)   # [S]  — anchor positions
    S          = len(stride_idx)
    K_s        = k[:, :, stride_idx, :].contiguous()      # [B, H, S, d]
    V_s        = v[:, :, stride_idx, :].contiguous()

    # ONE bmm for all T queries vs S anchor keys → single MKL DGEMM.
    scores_s_full = torch.bmm(
        q.reshape(BH, T, d),
        K_s.reshape(BH, S, d).transpose(1, 2),
    ).view(B, H_q, T, S) * scale                          # [B, H, T, S]

    # Strided causal mask: anchor at position stride_idx[s] is visible to
    # query at position t only when stride_idx[s] < t - sqrt_n + 1
    # (i.e. anchor is outside / before the local window — prevents double-count).
    mask_s = stride_idx[None, :] < (t_idx[:, None] - sqrt_n + 1)  # [T, S]
    scores_s_full.masked_fill_(~mask_s[None, None], float('-inf'))

    # Phase 3: relay block-mean keys.
    if use_relay:
        num_blocks = (T + sqrt_n - 1) // sqrt_n
        pad        = num_blocks * sqrt_n - T
        k_pad      = F.pad(k, (0, 0, 0, pad)) if pad > 0 else k
        v_pad      = F.pad(v, (0, 0, 0, pad)) if pad > 0 else v
        K_relay    = k_pad.view(B, H_q, num_blocks, sqrt_n, d).mean(dim=3).contiguous()
        V_relay    = v_pad.view(B, H_q, num_blocks, sqrt_n, d).mean(dim=3).contiguous()

        scores_r_full = torch.bmm(
            q.reshape(BH, T, d),
            K_relay.reshape(BH, num_blocks, d).transpose(1, 2),
        ).view(B, H_q, T, num_blocks) * scale             # [B, H, T, NB]

        b_ends = (torch.arange(num_blocks, device=dev) + 1) * sqrt_n - 1  # [NB]
        mask_r = b_ends[None, :] < (t_idx[:, None] - sqrt_n + 1)          # [T, NB]
        scores_r_full.masked_fill_(~mask_r[None, None], float('-inf'))
    else:
        num_blocks = 0
        K_relay = V_relay = scores_r_full = None

    # ── Phase-1 local window — chunk loop (K_w changes per chunk) ───────────
    # chunk_size=512: per-chunk working set ≈ 20 MB → stays in L3.
    # Larger values (≥1024) allocate >100 MB intermediates and hurt badly.
    # Phases 2+3 slices (read sequentially from L3-resident pre-computed tensors).
    output = torch.zeros(B, H_q, T, d, device=dev, dtype=q.dtype)

    for q_base in range(0, T, chunk_size):
        q_end = min(q_base + chunk_size, T)
        C     = q_end - q_base
        q_c   = q[:, :, q_base:q_end, :]                 # [B, H, C, d]

        # Local window: [q_base - sqrt_n + 1, q_end) — always contiguous in k
        win_s  = max(0, q_base - sqrt_n + 1)
        W      = q_end - win_s
        K_w    = k[:, :, win_s:q_end, :]
        V_w    = v[:, :, win_s:q_end, :]

        scores_l = torch.bmm(
            q_c.reshape(BH, C, d),
            K_w.reshape(BH, W, d).transpose(1, 2),
        ).view(B, H_q, C, W) * scale

        q_pos  = t_idx[q_base:q_end]
        k_pos  = t_idx[win_s:q_end]
        mask_l = (k_pos[None, :] <= q_pos[:, None]) & \
                 (k_pos[None, :] >= q_pos[:, None] - sqrt_n + 1)
        scores_l.masked_fill_(~mask_l[None, None], float('-inf'))

        # Pull pre-computed Phase 2+3 slices for this chunk (sequential L3 read)
        sc_s = scores_s_full[:, :, q_base:q_end, :]      # [B, H, C, S]

        if use_relay:
            sc_r       = scores_r_full[:, :, q_base:q_end, :]
            all_scores = torch.cat([scores_l, sc_s, sc_r], dim=-1)
        else:
            all_scores = torch.cat([scores_l, sc_s], dim=-1)

        attn  = torch.softmax(all_scores, dim=-1)
        a_l   = attn[:, :, :, :W]
        a_s   = attn[:, :, :, W:W+S]

        out_c = torch.bmm(a_l.reshape(BH, C, W),  V_w.reshape(BH, W,  d))
        out_c = out_c + torch.bmm(a_s.reshape(BH, C, S), V_s.reshape(BH, S, d))

        if use_relay:
            a_r   = attn[:, :, :, W+S:]
            out_c = out_c + torch.bmm(a_r.reshape(BH, C, num_blocks),
                                      V_relay.reshape(BH, num_blocks, d))

        output[:, :, q_base:q_end, :] = out_c.view(B, H_q, C, d)

    return output

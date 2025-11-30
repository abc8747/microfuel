"""Vendored and patched versions of `flash-linear-attention`.

To avoid recompilation during variable-length training, we make the
following changes:

1. Causal Conv1D (https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/convolution.py)

- `NB` removed from autotune key
- `NB` removed from constexpr list in signature (kept as scalar)
- `BT` fixed to 64 in wrappers

2. L2Norm (https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py)

- `NB` removed from autotune key
- `NB` removed from constexpr in signature
- `T` removed from constexpr in signature

3. GatedNorm (https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/fused_norm_gate.py)

- `NB` removed from autotune key
- `NB` removed from constexpr in signature
"""

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# See MIT License: https://github.com/fla-org/flash-linear-attention/blob/main/LICENSE
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from einops import rearrange
from fla.modules import convolution, fused_norm_gate, l2norm
from fla.ops.utils import prepare_chunk_indices
from fla.utils import autotune_cache_kwargs, get_multiprocessor_count, input_guard, is_amd

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [4, 8, 16, 32]
FIXED_BT_CONV = 64  # fixed block size for convolution to prevent dynamic resizing based on T


# fmt: off
@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D', 'W'],  # removed NB
    **autotune_cache_kwargs,
)
@triton.jit
def causal_conv1d_fwd_kernel(
    x, y, weight, bias, residual, cu_seqlens, initial_state, chunk_indices,
    B, T,
    D: tl.constexpr, W: tl.constexpr,
    BT: tl.constexpr, BW: tl.constexpr, BD: tl.constexpr,
    NB,  # removed constexpr
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    if HAS_WEIGHT:
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)

    b_y = tl.zeros((BT, BD), dtype=tl.float32)
    if not USE_INITIAL_STATE:
        for i_w in tl.static_range(-W + 1, 1):
            p_yi = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            b_yi = tl.load(p_yi, boundary_check=(0, 1)).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    elif i_t * BT >= W:
        for i_w in tl.static_range(-W + 1, 1):
            p_yi = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            b_yi = tl.load(p_yi, boundary_check=(0, 1)).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    else:
        o_t = i_t * BT + tl.arange(0, BT)
        for i_w in tl.static_range(-W + 1, 1):
            o_x = o_t + i_w
            m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d
            m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d
            b_yi = tl.load(x + bos * D + o_x[:, None] * D + o_d, mask=m_x, other=0).to(tl.float32)
            b_yi += tl.load(initial_state + i_n * D*W + o_d * W + (o_x + W)[:, None], mask=m_c, other=0).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)
    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * tl.sigmoid(b_y)
    if HAS_RESIDUAL:
        p_residual = tl.make_block_ptr(residual + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        b_residual = tl.load(p_residual, boundary_check=(0, 1))
        b_y += b_residual

    p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['dw'] is not None,
    'HAS_BIAS': lambda args: args['db'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in [4, 8, 16, 32]
    ],
    key=['D', 'W'],  # removed NB
    **autotune_cache_kwargs,
)
@triton.jit
def causal_conv1d_bwd_kernel(
    x, y, weight, initial_state, dh0, dht, dy, dx, dw, db, cu_seqlens, chunk_indices,
    B, T,
    D: tl.constexpr, W: tl.constexpr,
    BT: tl.constexpr, BW: tl.constexpr, BD: tl.constexpr,
    NB,  # removed constexpr
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, USE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_tg = i_b * tl.num_programs(1) + i_t
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    if HAS_WEIGHT:
        p_x = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        b_x = tl.load(p_x, boundary_check=(0, 1))
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0)

    b_dx = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_BIAS:
        b_db = tl.zeros((BD,), dtype=tl.float32)

    if not USE_FINAL_STATE:
        for i_w in tl.static_range(0, W):
            p_dy = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
                b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                b_ys = tl.sigmoid(b_y)
                b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
            b_wdy = b_dy
            if HAS_WEIGHT:
                b_wdy = b_wdy * tl.sum(b_w * (o_w == (W - i_w - 1)), 1)
                b_dw = tl.sum(b_dy * b_x, 0)
                tl.store(dw + i_tg * D*W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)
            if HAS_BIAS and i_w == 0:
                b_db += tl.sum(b_dy, 0)
            b_dx += b_wdy
    elif i_t * BT >= W:
        for i_w in tl.static_range(0, W):
            p_dy = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
                b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                b_ys = tl.sigmoid(b_y)
                b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
            b_wdy = b_dy
            if HAS_WEIGHT:
                b_wdy = b_wdy * tl.sum(b_w * (o_w == (W - i_w - 1)), 1)
                b_dw = tl.sum(b_dy * b_x, 0)
                tl.store(dw + i_tg * D*W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)
            if HAS_BIAS and i_w == 0:
                b_db += tl.sum(b_dy, 0)
            b_dx += b_wdy
    else:
        o_t = i_t * BT + tl.arange(0, BT)
        for i_w in tl.static_range(0, W):
            p_dy = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            b_dy_shift = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
                b_y_shift = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                b_ys = tl.sigmoid(b_y_shift)
                b_dy_shift = b_dy_shift * b_ys * (1 + b_y_shift * (1 - b_ys))
            if HAS_WEIGHT:
                b_dw = tl.sum(b_dy_shift * b_x, 0)
                if USE_INITIAL_STATE:
                    mask_head_rows = (o_t < i_w)
                    b_dy_head = tl.load(dy + bos * D + o_t[:, None] * D + o_d, mask=(mask_head_rows[:, None] & m_d[None, :]), other=0.0).to(tl.float32)
                    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                        b_y_head = tl.load(y + bos * D + o_t[:, None] * D + o_d, mask=(mask_head_rows[:, None] & m_d[None, :]), other=0.0).to(tl.float32)
                        b_ys_head = tl.sigmoid(b_y_head)
                        b_dy_head = b_dy_head * b_ys_head * (1 + b_y_head * (1 - b_ys_head))
                    o_c = W - i_w + o_t
                    mask_c = (mask_head_rows & (o_c >= 1) & (o_c < W))
                    b_xc = tl.load(initial_state + i_n * D * W + o_d[None, :] * W + o_c[:, None], mask=(mask_c[:, None] & m_d[None, :]), other=0.0).to(tl.float32)
                    b_dw += tl.sum(b_dy_head * b_xc, 0)
                tl.store(dw + i_tg * D * W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)

            if HAS_BIAS and i_w == 0:
                b_db += tl.sum(b_dy_shift, 0)
            b_wdy = b_dy_shift if not HAS_WEIGHT else (b_dy_shift * tl.sum(b_w * (o_w == (W - i_w - 1)), 1))
            b_dx += b_wdy

        if USE_INITIAL_STATE:
            p_dy0 = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
            b_dy0 = tl.load(p_dy0, boundary_check=(0, 1)).to(tl.float32)
            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                p_y0 = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
                b_y0 = tl.load(p_y0, boundary_check=(0, 1)).to(tl.float32)
                b_ys0 = tl.sigmoid(b_y0)
                b_dy0 = b_dy0 * b_ys0 * (1 + b_y0 * (1 - b_ys0))
            for i_w in tl.static_range(1, W):
                m_rows = (o_t < i_w)
                if HAS_WEIGHT:
                    w_idx_rows = i_w - 1 - o_t
                    w_mask = (o_w[None, :] == w_idx_rows[:, None])
                    w_pick = tl.sum(b_w[None, :, :] * w_mask[:, None, :], 2)
                else:
                    w_pick = 1.0
                contrib = (b_dy0 * w_pick).to(tl.float32)
                contrib = tl.where(m_rows[:, None] & m_d[None, :], contrib, 0.0)
                b_dh0_s = tl.sum(contrib, 0)
                tl.store(dh0 + i_t * B * D * W + i_n * D * W + o_d * W + i_w, b_dh0_s.to(dh0.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_d)

    if HAS_BIAS:
        b_db = tl.cast(b_db, dtype=db.dtype.element_ty, fp_downcast_rounding='rtne')
        tl.store(db + i_tg * D + o_d, b_db, mask=m_d)

    if USE_FINAL_STATE:
        if i_t * BT + BT >= T-W:
            start_tok = max(0, T - (W - 1))
            offset = i_t * BT + tl.arange(0, BT)
            tok_idx = offset - start_tok
            mask = (offset >= start_tok) & (offset < T)
            w_idx = 1 + tok_idx
            dht_off = i_n * D * W + o_d[None, :] * W + w_idx[:, None]
            b_dht = tl.load(dht + dht_off, mask=mask[:, None] & m_d[None, :], other=0.).to(tl.float32)
            b_dx += b_dht

    p_dx = tl.make_block_ptr(dx + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    tl.store(p_dx, tl.cast(b_dx, dtype=p_dx.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))


@input_guard
def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')
    B, T, D, W = *x.shape, weight.shape[1]

    # HACK: fix BT to constant to avoid recompile on varlen batches
    BT = FIXED_BT_CONV
    BW = triton.next_power_of_2(W)
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    NB = triton.cdiv(B*T, 1024)

    y = torch.empty_like(x)
    def grid(meta): return (triton.cdiv(D, meta['BD']), NT, B)
    causal_conv1d_fwd_kernel[grid](
        x=x, y=y, weight=weight, bias=bias, residual=residual,
        cu_seqlens=cu_seqlens, initial_state=initial_state, chunk_indices=chunk_indices,
        B=B, T=T, D=D, W=W, BT=BT, BW=BW, NB=NB,
        ACTIVATION=activation,
    )
    final_state = None
    if output_final_state:
        # NOTE: we use the original util since it doesn't depend on the kernel modifications
        final_state = convolution.causal_conv1d_update_states(
            x=x, state_len=W, initial_state=initial_state, cu_seqlens=cu_seqlens,
        )
    return y.view(shape), final_state


def causal_conv1d_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    dht: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
):
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')
    B, T, D = x.shape
    W = weight.shape[1] if weight is not None else None

    # HACK: fix BT
    BT = FIXED_BT_CONV
    BW = triton.next_power_of_2(W)
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    NB = triton.cdiv(B*T, 1024)

    y = None
    if activation is not None:
        y, _ = causal_conv1d_fwd(
            x=x, weight=weight, bias=bias, residual=None, initial_state=initial_state,
            activation=None, cu_seqlens=cu_seqlens, output_final_state=False,
        )
    dx = torch.empty_like(x)
    dw = weight.new_empty(B*NT, *weight.shape, dtype=torch.float) if weight is not None else None
    db = bias.new_empty(B*NT, *bias.shape, dtype=torch.float) if bias is not None else None
    dr = dy if residual is not None else None
    dh0 = initial_state.new_zeros(min(NT, triton.cdiv(W, BT)), *initial_state.shape) if initial_state is not None else None

    def grid(meta): return (triton.cdiv(D, meta['BD']), NT, B)
    causal_conv1d_bwd_kernel[grid](
        x=x, y=y, weight=weight, initial_state=initial_state, dh0=dh0, dht=dht,
        dy=dy, dx=dx, dw=dw, db=db, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        B=B, T=T, D=D, W=W, BT=BT, BW=BW, NB=NB,
        ACTIVATION=activation,
    )
    if weight is not None:
        dw = dw.sum(0).to(weight)
    if bias is not None:
        db = db.sum(0).to(bias)
    if initial_state is not None:
        dh0 = dh0.sum(0, dtype=torch.float32).to(initial_state)

    return dx.view(shape), dw, db, dr, dh0


BT_LIST = [8, 16, 32, 64, 128]

@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
        for BT in BT_LIST
    ],
    key=['D'],  # NB removed
    **autotune_cache_kwargs,
)
@triton.jit
def l2norm_fwd_kernel(
    x, y, rstd, eps,
    T,  # removed constexpr
    D: tl.constexpr, BD: tl.constexpr, 
    NB,  # removed constexpr
    BT: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))

    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    b_y = b_x * b_rstd[:, None]

    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))

@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
        for BT in BT_LIST
    ],
    key=['D'],  # NB removed
    **autotune_cache_kwargs,
)
@triton.jit
def l2norm_bwd_kernel(
    y, rstd, dy, dx, eps,
    T,  # removed constexpr
    D: tl.constexpr, BD: tl.constexpr,
    NB,  # removed constexpr
    BT: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    p_dy = tl.make_block_ptr(dy, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_dx = tl.make_block_ptr(dx, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))

    b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
    b_rstd = tl.load(p_rstd, boundary_check=(0,)).to(tl.float32)
    b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
    b_dx = b_dy * b_rstd[:, None] - tl.sum(b_dy * b_y, 1)[:, None] * b_y * b_rstd[:, None]
    tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    if D <= 512:
        NB = triton.cdiv(T, 2048)
        def grid(meta): return (triton.cdiv(T, meta['BT']), )
        l2norm_fwd_kernel[grid](
            x=x, y=y, rstd=rstd, eps=eps, T=T, D=D, BD=BD, NB=NB,
        )
    else:
        # fallback to original kernel1 for large D (no NB/T constexpr issues there)
        l2norm.l2norm_fwd_kernel1[(T,)](
            x=x, y=y, rstd=rstd, eps=eps, D=D, BD=BD,
        )
    return y.view(x_shape_og), rstd.view(x_shape_og[:-1])


def l2norm_bwd(
    y: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-6,
):
    y_shape_og = y.shape
    y = y.view(-1, dy.shape[-1])
    dy = dy.view(-1, dy.shape[-1])
    assert dy.shape == y.shape
    dx = torch.empty_like(y)
    T, D = y.shape[0], y.shape[-1]
    MAX_FUSED_SIZE = 65536 // y.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    if D <= 512:
        NB = triton.cdiv(T, 2048)
        def grid(meta): return (triton.cdiv(T, meta['BT']), )
        l2norm_bwd_kernel[grid](
            y=y, rstd=rstd, dy=dy, dx=dx, eps=eps, T=T, D=D, BD=BD, NB=NB,
        )
    else:
        l2norm.l2norm_bwd_kernel1[(T,)](
            y=y, rstd=rstd, dy=dy, dx=dx, eps=eps, D=D, BD=BD,
        )

    return dx.view(y_shape_og)


@triton.heuristics({
    'STORE_RESIDUAL_OUT': lambda args: args['residual_out'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'HAS_WEIGHT': lambda args: args['w'] is not None,
    'HAS_BIAS': lambda args: args['b'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for num_warps in [4, 8, 16]
    ],
    key=['D', 'IS_RMS_NORM', 'STORE_RESIDUAL_OUT', 'HAS_RESIDUAL', 'HAS_WEIGHT'],  # NB removed
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_gated_fwd_kernel(
    x, g, y, w, b, residual, residual_out, mean, rstd, eps,
    T,
    D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
    NB,  # removed constexpr
    ACTIVATION: tl.constexpr, IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0)
    o_d = tl.arange(0, BD)
    m_d = o_d < D

    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    if HAS_RESIDUAL:
        p_res = tl.make_block_ptr(residual, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
        b_x += tl.load(p_res, boundary_check=(0, 1)).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        p_res_out = tl.make_block_ptr(residual_out, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
        tl.store(p_res_out, b_x.to(p_res_out.dtype.element_ty), boundary_check=(0, 1))
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=1) / D
        p_mean = tl.make_block_ptr(mean, (T,), (1,), (i_t * BT,), (BT,), (0,))
        tl.store(p_mean, b_mean.to(p_mean.dtype.element_ty), boundary_check=(0,))
        b_xbar = tl.where(m_d[None, :], b_x - b_mean[:, None], 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    else:
        b_xbar = tl.where(m_d[None, :], b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (b_x - b_mean[:, None]) * b_rstd[:, None] if not IS_RMS_NORM else b_x * b_rstd[:, None]
    b_y = b_x_hat * b_w[None, :] if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b[None, :]

    p_g = tl.make_block_ptr(g, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == 'sigmoid':
        b_y = b_y * tl.sigmoid(b_g)

    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'HAS_DRESIDUAL': lambda args: args['dresidual'] is not None,
    'HAS_WEIGHT': lambda args: args['w'] is not None,
    'HAS_BIAS': lambda args: args['b'] is not None,
    'RECOMPUTE_OUTPUT': lambda args: args['y'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for num_warps in [4, 8, 16]
    ],
    key=['D', 'IS_RMS_NORM', 'HAS_DRESIDUAL', 'HAS_WEIGHT'],  # NB removed
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_gated_bwd_kernel(
    x, g, w, b, y, dy, dx, dg, dw, db, dresidual, dresidual_in, mean, rstd,
    T, BS,
    D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
    NB,  # removed constexpr
    ACTIVATION: tl.constexpr, IS_RMS_NORM: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr, HAS_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr, RECOMPUTE_OUTPUT: tl.constexpr,
):
    i_s = tl.program_id(0)
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
        b_dw = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d, other=0.0).to(tl.float32)
        b_db = tl.zeros((BT, BD), dtype=tl.float32)

    T = min(i_s * BS + BS, T)
    for i_t in range(i_s * BS, T, BT):
        p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
        p_g = tl.make_block_ptr(g, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
        p_dy = tl.make_block_ptr(dy, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
        p_dx = tl.make_block_ptr(dx, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
        p_dg = tl.make_block_ptr(dg, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
        b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)

        if not IS_RMS_NORM:
            p_mean = tl.make_block_ptr(mean, (T,), (1,), (i_t,), (BT,), (0,))
            b_mean = tl.load(p_mean, boundary_check=(0,))
        p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t,), (BT,), (0,))
        b_rstd = tl.load(p_rstd, boundary_check=(0,))
        b_xhat = (b_x - b_mean[:, None]) * b_rstd[:, None] if not IS_RMS_NORM else b_x * b_rstd[:, None]
        b_xhat = tl.where(m_d[None, :], b_xhat, 0.0)

        b_y = b_xhat * b_w[None, :] if HAS_WEIGHT else b_xhat
        if HAS_BIAS:
            b_y = b_y + b_b[None, :]
        if RECOMPUTE_OUTPUT:
            p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
            tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))

        b_sigmoid_g = tl.sigmoid(b_g)
        if ACTIVATION == 'swish' or ACTIVATION == 'silu':
            b_dg = b_dy * b_y * (b_sigmoid_g + b_g * b_sigmoid_g * (1 - b_sigmoid_g))
            b_dy = b_dy * b_g * b_sigmoid_g
        elif ACTIVATION == 'sigmoid':
            b_dg = b_dy * b_y * b_sigmoid_g * (1 - b_sigmoid_g)
            b_dy = b_dy * b_sigmoid_g
        b_wdy = b_dy

        if HAS_WEIGHT or HAS_BIAS:
            m_t = (i_t + tl.arange(0, BT)) < T
        if HAS_WEIGHT:
            b_wdy = b_dy * b_w
            b_dw += tl.where(m_t[:, None], b_dy * b_xhat, 0.0)
        if HAS_BIAS:
            b_db += tl.where(m_t[:, None], b_dy, 0.0)
        if not IS_RMS_NORM:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=1) / D
            b_c2 = tl.sum(b_wdy, axis=1) / D
            b_dx = (b_wdy - (b_xhat * b_c1[:, None] + b_c2[:, None])) * b_rstd[:, None]
        else:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=1) / D
            b_dx = (b_wdy - b_xhat * b_c1[:, None]) * b_rstd[:, None]
        if HAS_DRESIDUAL:
            p_dres = tl.make_block_ptr(dresidual, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
            b_dres = tl.load(p_dres, boundary_check=(0, 1)).to(tl.float32)
            b_dx += b_dres
        if STORE_DRESIDUAL:
            p_dres_in = tl.make_block_ptr(dresidual_in, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
            tl.store(p_dres_in, b_dx.to(p_dres_in.dtype.element_ty), boundary_check=(0, 1))

        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))

    if HAS_WEIGHT:
        tl.store(dw + i_s * D + o_d, tl.sum(b_dw, axis=0), mask=m_d)
    if HAS_BIAS:
        tl.store(db + i_s * D + o_d, tl.sum(b_db, axis=0), mask=m_d)


def layer_norm_gated_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = 'swish',
    eps: float = 1e-5,
    residual: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    residual_dtype: torch.dtype = None,
    is_rms_norm: bool = False,
):
    if residual is not None:
        residual_dtype = residual.dtype
    T, D = x.shape
    if residual is not None:
        assert residual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (D,)
    if bias is not None:
        assert bias.shape == (D,)
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = torch.empty((T,), dtype=torch.float, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    if D <= 512:
        NB = triton.cdiv(T, 2048)
        def grid(meta): return (triton.cdiv(T, meta['BT']),)
        layer_norm_gated_fwd_kernel[grid](
            x=x, g=g, y=y, w=weight, b=bias, residual=residual, residual_out=residual_out,
            mean=mean, rstd=rstd, eps=eps, T=T, D=D, BD=BD, NB=NB,
            ACTIVATION=activation, IS_RMS_NORM=is_rms_norm,
        )
    else:
        fused_norm_gate.layer_norm_gated_fwd_kernel1[(T,)](
            x=x, g=g, y=y, w=weight, b=bias, residual=residual, residual_out=residual_out,
            mean=mean, rstd=rstd, eps=eps, D=D, BD=BD,
            ACTIVATION=activation, IS_RMS_NORM=is_rms_norm,
        )
    return y, mean, rstd, residual_out if residual_out is not None else x


def layer_norm_gated_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = 'swish',
    eps: float = 1e-5,
    mean: torch.Tensor = None,
    rstd: torch.Tensor = None,
    dresidual: torch.Tensor = None,
    has_residual: bool = False,
    is_rms_norm: bool = False,
    x_dtype: torch.dtype = None,
    recompute_output: bool = False,
):
    T, D = x.shape
    dx = torch.empty_like(x) if x_dtype is None else torch.empty(T, D, dtype=x_dtype, device=x.device)
    dg = torch.empty_like(g) if x_dtype is None else torch.empty(T, D, dtype=x_dtype, device=x.device)
    dresidual_in = torch.empty_like(x) if has_residual and dx.dtype != x.dtype else None
    y = torch.empty(T, D, dtype=dy.dtype, device=dy.device) if recompute_output else None

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    NS = get_multiprocessor_count(x.device.index)
    BS = math.ceil(T / NS)

    dw = torch.empty((NS, D), dtype=torch.float, device=weight.device) if weight is not None else None
    db = torch.empty((NS, D), dtype=torch.float, device=bias.device) if bias is not None else None
    grid = (NS,)

    if D <= 512:
        NB = triton.cdiv(T, 2048)
        layer_norm_gated_bwd_kernel[grid](
            x=x, g=g, w=weight, b=bias, y=y, dy=dy, dx=dx, dg=dg, dw=dw, db=db,
            dresidual=dresidual, dresidual_in=dresidual_in, mean=mean, rstd=rstd,
            T=T, D=D, BS=BS, BD=BD, NB=NB,
            ACTIVATION=activation, IS_RMS_NORM=is_rms_norm,
            STORE_DRESIDUAL=dresidual_in is not None,
        )
    else:
        fused_norm_gate.layer_norm_gated_bwd_kernel1[grid](
            x=x, g=g, w=weight, b=bias, y=y, dy=dy, dx=dx, dg=dg, dw=dw, db=db,
            dresidual=dresidual, dresidual_in=dresidual_in, mean=mean, rstd=rstd,
            T=T, D=D, BS=BS, BD=BD,
            ACTIVATION=activation, IS_RMS_NORM=is_rms_norm,
            STORE_DRESIDUAL=dresidual_in is not None,
        )
    dw = dw.sum(0).to(weight.dtype) if weight is not None else None
    db = db.sum(0).to(bias.dtype) if bias is not None else None
    if has_residual and dx.dtype == x.dtype:
        dresidual_in = dx
    return (dx, dg, dw, db, dresidual_in) if not recompute_output else (dx, dg, dw, db, dresidual_in, y)

# fmt: on


def install_optimized_kernels_():
    """Patches FLA modules with optimized versions to reduce JIT recompilation."""
    convolution.causal_conv1d_fwd_kernel = causal_conv1d_fwd_kernel
    convolution.causal_conv1d_bwd_kernel = causal_conv1d_bwd_kernel
    convolution.causal_conv1d_fwd = causal_conv1d_fwd
    convolution.causal_conv1d_bwd = causal_conv1d_bwd

    l2norm.l2norm_fwd_kernel = l2norm_fwd_kernel
    l2norm.l2norm_bwd_kernel = l2norm_bwd_kernel
    l2norm.l2norm_fwd = l2norm_fwd
    l2norm.l2norm_bwd = l2norm_bwd

    fused_norm_gate.layer_norm_gated_fwd_kernel = layer_norm_gated_fwd_kernel
    fused_norm_gate.layer_norm_gated_bwd_kernel = layer_norm_gated_bwd_kernel
    fused_norm_gate.layer_norm_gated_fwd = layer_norm_gated_fwd
    fused_norm_gate.layer_norm_gated_bwd = layer_norm_gated_bwd

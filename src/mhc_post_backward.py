"""FlagGems-style Triton backward for mhc_post.

Given the forward
    output[b,s,i,d] = sum_j h_res[b,s,j,i] * x[b,s,j,d]
                    + h_post[b,s,i] * h_out[b,s,d]

the gradients are:
    grad_x[b,s,j,d]      = sum_i  h_res[b,s,j,i] * grad_y[b,s,i,d]
    grad_h_res[b,s,j,i]  = sum_d  x[b,s,j,d]     * grad_y[b,s,i,d]
    grad_h_out[b,s,d]    = sum_i  h_post[b,s,i]  * grad_y[b,s,i,d]
    grad_h_post[b,s,i]   = sum_d  h_out[b,s,d]   * grad_y[b,s,i,d]

Ascend-oriented design
----------------------
1. Kernel 1 (`_mhc_post_bwd_dxdho`): produces `grad_x` (bf16/fp16) and
   `grad_h_out` (bf16/fp16). Grid = (T, cdiv(D, BLOCK_D)). All D-loop work
   for one token fits in one program: we load `h_res` and `h_post` once
   per token (small scalars, GM-cached), then loop D-tiles.
2. Kernel 2 (`_mhc_post_bwd_dresdpost`): produces `grad_h_res` (fp32) and
   `grad_h_post` (fp32) via a D-reduction. Uses `tl.atomic_add` on the
   two (T, n, n) and (T, n) outputs so multiple D-blocks per token can run
   concurrently. Grid = (T, cdiv(D, BLOCK_D)).
3. Unified single-pass over grad_y in each kernel — no triple reads of the
   grad tensor. Matches the "fuse the 3 passes" recommendation for AscendC.
4. HC_MULT=4 fast paths for both kernels: unroll `n=4` completely so each
   accumulator stays in registers and the compiler emits FMA chains.
5. Autotune configs: shallow num_stages (1..2), moderate BLOCK_D.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kernel 1: grad_x and grad_h_out (HC_MULT=4)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 1024}, num_warps=8, num_stages=1),
    ],
    key=["D"],
)
@triton.jit
def _mhc_post_bwd_dxdho_hc4(
    grad_y_ptr,     # (T, 4, D)  bf16/fp16
    h_res_ptr,      # (T, 4, 4)  fp32
    h_post_ptr,     # (T, 4)     fp32
    grad_x_ptr,     # (T, 4, D)  bf16/fp16   OUTPUT
    grad_hout_ptr,  # (T, D)     bf16/fp16   OUTPUT
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    gy_base = pid_t * 4 * D
    gx_base = pid_t * 4 * D
    gho_base = pid_t * D
    hres_base = pid_t * 16
    hpost_base = pid_t * 4

    r00 = tl.load(h_res_ptr + hres_base + 0)
    r01 = tl.load(h_res_ptr + hres_base + 1)
    r02 = tl.load(h_res_ptr + hres_base + 2)
    r03 = tl.load(h_res_ptr + hres_base + 3)
    r10 = tl.load(h_res_ptr + hres_base + 4)
    r11 = tl.load(h_res_ptr + hres_base + 5)
    r12 = tl.load(h_res_ptr + hres_base + 6)
    r13 = tl.load(h_res_ptr + hres_base + 7)
    r20 = tl.load(h_res_ptr + hres_base + 8)
    r21 = tl.load(h_res_ptr + hres_base + 9)
    r22 = tl.load(h_res_ptr + hres_base + 10)
    r23 = tl.load(h_res_ptr + hres_base + 11)
    r30 = tl.load(h_res_ptr + hres_base + 12)
    r31 = tl.load(h_res_ptr + hres_base + 13)
    r32 = tl.load(h_res_ptr + hres_base + 14)
    r33 = tl.load(h_res_ptr + hres_base + 15)

    hp0 = tl.load(h_post_ptr + hpost_base + 0)
    hp1 = tl.load(h_post_ptr + hpost_base + 1)
    hp2 = tl.load(h_post_ptr + hpost_base + 2)
    hp3 = tl.load(h_post_ptr + hpost_base + 3)

    gy0 = tl.load(grad_y_ptr + gy_base + 0 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    gy1 = tl.load(grad_y_ptr + gy_base + 1 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    gy2 = tl.load(grad_y_ptr + gy_base + 2 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    gy3 = tl.load(grad_y_ptr + gy_base + 3 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)

    # grad_x[j] = sum_i h_res[j,i] * grad_y[i]
    gx0 = r00 * gy0 + r01 * gy1 + r02 * gy2 + r03 * gy3
    gx1 = r10 * gy0 + r11 * gy1 + r12 * gy2 + r13 * gy3
    gx2 = r20 * gy0 + r21 * gy1 + r22 * gy2 + r23 * gy3
    gx3 = r30 * gy0 + r31 * gy1 + r32 * gy2 + r33 * gy3

    # grad_h_out = sum_i h_post[i] * grad_y[i]
    gho = hp0 * gy0 + hp1 * gy1 + hp2 * gy2 + hp3 * gy3

    dt = grad_x_ptr.dtype.element_ty
    tl.store(grad_x_ptr + gx_base + 0 * D + d_off, gx0.to(dt), mask=d_mask)
    tl.store(grad_x_ptr + gx_base + 1 * D + d_off, gx1.to(dt), mask=d_mask)
    tl.store(grad_x_ptr + gx_base + 2 * D + d_off, gx2.to(dt), mask=d_mask)
    tl.store(grad_x_ptr + gx_base + 3 * D + d_off, gx3.to(dt), mask=d_mask)
    tl.store(grad_hout_ptr + gho_base + d_off, gho.to(dt), mask=d_mask)


# ---------------------------------------------------------------------------
# Kernel 2: grad_h_res and grad_h_post (HC_MULT=4)
#   One program per token (pid_t). Loops over D in BLOCK_D chunks and
#   accumulates the 16 (grad_h_res) + 4 (grad_h_post) partial sums in
#   registers (Vector core Unified Buffer), then writes them once at the
#   end. This avoids atomic_add contention on Global Memory and makes the
#   reduction deterministic.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 1024}, num_warps=8, num_stages=1),
    ],
    key=["D"],
)
@triton.jit
def _mhc_post_bwd_dresdpost_hc4(
    grad_y_ptr,      # (T, 4, D)  bf16/fp16
    x_ptr,           # (T, 4, D)  bf16/fp16
    h_out_ptr,       # (T, D)     bf16/fp16
    grad_hres_ptr,   # (T, 4, 4)  fp32     (plain store, no atomic)
    grad_hpost_ptr,  # (T, 4)     fp32     (plain store, no atomic)
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)

    gy_base = pid_t * 4 * D
    x_base = pid_t * 4 * D
    hout_base = pid_t * D

    # fp32 register accumulators (initialized to zero)
    p0 = tl.zeros([], dtype=tl.float32)
    p1 = tl.zeros([], dtype=tl.float32)
    p2 = tl.zeros([], dtype=tl.float32)
    p3 = tl.zeros([], dtype=tl.float32)
    r00 = tl.zeros([], dtype=tl.float32); r01 = tl.zeros([], dtype=tl.float32)
    r02 = tl.zeros([], dtype=tl.float32); r03 = tl.zeros([], dtype=tl.float32)
    r10 = tl.zeros([], dtype=tl.float32); r11 = tl.zeros([], dtype=tl.float32)
    r12 = tl.zeros([], dtype=tl.float32); r13 = tl.zeros([], dtype=tl.float32)
    r20 = tl.zeros([], dtype=tl.float32); r21 = tl.zeros([], dtype=tl.float32)
    r22 = tl.zeros([], dtype=tl.float32); r23 = tl.zeros([], dtype=tl.float32)
    r30 = tl.zeros([], dtype=tl.float32); r31 = tl.zeros([], dtype=tl.float32)
    r32 = tl.zeros([], dtype=tl.float32); r33 = tl.zeros([], dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_off = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_off < D

        gy0 = tl.load(grad_y_ptr + gy_base + 0 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
        gy1 = tl.load(grad_y_ptr + gy_base + 1 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
        gy2 = tl.load(grad_y_ptr + gy_base + 2 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
        gy3 = tl.load(grad_y_ptr + gy_base + 3 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)

        x0 = tl.load(x_ptr + x_base + 0 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + x_base + 1 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + x_base + 2 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + x_base + 3 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)

        ho = tl.load(h_out_ptr + hout_base + d_off, mask=d_mask, other=0.0).to(tl.float32)

        p0 += tl.sum(ho * gy0, axis=0)
        p1 += tl.sum(ho * gy1, axis=0)
        p2 += tl.sum(ho * gy2, axis=0)
        p3 += tl.sum(ho * gy3, axis=0)

        r00 += tl.sum(x0 * gy0, axis=0); r01 += tl.sum(x0 * gy1, axis=0)
        r02 += tl.sum(x0 * gy2, axis=0); r03 += tl.sum(x0 * gy3, axis=0)
        r10 += tl.sum(x1 * gy0, axis=0); r11 += tl.sum(x1 * gy1, axis=0)
        r12 += tl.sum(x1 * gy2, axis=0); r13 += tl.sum(x1 * gy3, axis=0)
        r20 += tl.sum(x2 * gy0, axis=0); r21 += tl.sum(x2 * gy1, axis=0)
        r22 += tl.sum(x2 * gy2, axis=0); r23 += tl.sum(x2 * gy3, axis=0)
        r30 += tl.sum(x3 * gy0, axis=0); r31 += tl.sum(x3 * gy1, axis=0)
        r32 += tl.sum(x3 * gy2, axis=0); r33 += tl.sum(x3 * gy3, axis=0)

    hres_base = pid_t * 16
    hpost_base = pid_t * 4

    tl.store(grad_hpost_ptr + hpost_base + 0, p0)
    tl.store(grad_hpost_ptr + hpost_base + 1, p1)
    tl.store(grad_hpost_ptr + hpost_base + 2, p2)
    tl.store(grad_hpost_ptr + hpost_base + 3, p3)

    tl.store(grad_hres_ptr + hres_base + 0, r00)
    tl.store(grad_hres_ptr + hres_base + 1, r01)
    tl.store(grad_hres_ptr + hres_base + 2, r02)
    tl.store(grad_hres_ptr + hres_base + 3, r03)
    tl.store(grad_hres_ptr + hres_base + 4, r10)
    tl.store(grad_hres_ptr + hres_base + 5, r11)
    tl.store(grad_hres_ptr + hres_base + 6, r12)
    tl.store(grad_hres_ptr + hres_base + 7, r13)
    tl.store(grad_hres_ptr + hres_base + 8, r20)
    tl.store(grad_hres_ptr + hres_base + 9, r21)
    tl.store(grad_hres_ptr + hres_base + 10, r22)
    tl.store(grad_hres_ptr + hres_base + 11, r23)
    tl.store(grad_hres_ptr + hres_base + 12, r30)
    tl.store(grad_hres_ptr + hres_base + 13, r31)
    tl.store(grad_hres_ptr + hres_base + 14, r32)
    tl.store(grad_hres_ptr + hres_base + 15, r33)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _flatten(grad_y, x, h_res, h_out, h_post):
    if grad_y.dim() == 4:
        B, S, N, D = grad_y.shape
        T = B * S
        shape4 = (B, S, N, D)
        shape_out = (B, S, D)
        return (
            grad_y.reshape(T, N, D).contiguous(),
            x.reshape(T, N, D).contiguous(),
            h_res.reshape(T, N, N).contiguous(),
            h_out.reshape(T, D).contiguous(),
            h_post.reshape(T, N).contiguous(),
            shape4,
            shape_out,
        )
    if grad_y.dim() == 3:
        T, N, D = grad_y.shape
        return (
            grad_y.contiguous(),
            x.contiguous(),
            h_res.reshape(T, N, N).contiguous(),
            h_out.reshape(T, D).contiguous(),
            h_post.reshape(T, N).contiguous(),
            (T, N, D),
            (T, D),
        )
    raise ValueError(f"unsupported grad_y.dim()={grad_y.dim()}")


def mhc_post_backward(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    h_res: torch.Tensor,
    h_out: torch.Tensor,
    h_post: torch.Tensor,
):
    """Compute (grad_x, grad_h_res, grad_h_out, grad_h_post)."""
    gy, xf, hres, hout, hpost, shape4, shape_out = _flatten(
        grad_y, x, h_res, h_out, h_post
    )
    T, N, D = gy.shape

    grad_x = torch.empty_like(xf)
    grad_hout = torch.empty_like(hout)
    # In-kernel reduction: written once per (t, i), no atomic needed.
    grad_hres = torch.empty_like(hres)
    grad_hpost = torch.empty_like(hpost)

    if N == 4:
        grid1 = lambda META: (T, triton.cdiv(D, META["BLOCK_D"]))
        _mhc_post_bwd_dxdho_hc4[grid1](
            gy, hres, hpost, grad_x, grad_hout, D=D,
        )
        # One program per token: loops over D internally and writes once.
        _mhc_post_bwd_dresdpost_hc4[(T,)](
            gy, xf, hout, grad_hres, grad_hpost, D=D,
        )
    else:
        # Torch fallback for arbitrary N.
        grad_x_ref, grad_hres_ref, grad_hout_ref, grad_hpost_ref = mhc_post_backward_ref(
            grad_y, x, h_res, h_out, h_post
        )
        return grad_x_ref, grad_hres_ref, grad_hout_ref, grad_hpost_ref

    return (
        grad_x.reshape(shape4),
        grad_hres.reshape(h_res.shape),
        grad_hout.reshape(shape_out),
        grad_hpost.reshape(h_post.shape),
    )


def mhc_post_backward_ref(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    h_res: torch.Tensor,
    h_out: torch.Tensor,
    h_post: torch.Tensor,
):
    """PyTorch reference for the four gradients."""
    orig_dtype = x.dtype
    gy, xf, hres, hout, hpost, shape4, shape_out = _flatten(
        grad_y, x, h_res, h_out, h_post
    )
    gy_f = gy.float()
    x_f = xf.float()
    ho_f = hout.float()

    # grad_x[j,d] = sum_i hres[j,i] * gy[i,d]
    grad_x = torch.einsum("tji,tid->tjd", hres, gy_f).to(orig_dtype)
    # grad_h_res[j,i] = sum_d x[j,d] * gy[i,d]
    grad_hres = torch.einsum("tjd,tid->tji", x_f, gy_f)
    # grad_h_out[d]  = sum_i hpost[i] * gy[i,d]
    grad_hout = torch.einsum("ti,tid->td", hpost, gy_f).to(orig_dtype)
    # grad_h_post[i] = sum_d hout[d] * gy[i,d]
    grad_hpost = torch.einsum("td,tid->ti", ho_f, gy_f)

    return (
        grad_x.reshape(shape4),
        grad_hres.reshape(h_res.shape),
        grad_hout.reshape(shape_out),
        grad_hpost.reshape(h_post.shape),
    )

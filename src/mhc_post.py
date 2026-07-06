"""FlagGems-style Triton implementation of the mhc_post operator.

Semantics:

    output[b,s,i,d] = sum_{j} h_res[b,s,j,i] * x[b,s,j,d]
                    + h_post[b,s,i] * h_out[b,s,d]

Layout supported (BSND / TND flattened to (T, n, D) with T = B*S):
    x:      (T, n, D)  bf16/fp16
    h_res:  (T, n, n)  fp32
    h_out:  (T, D)     bf16/fp16
    h_post: (T, n)     fp32
    output: (T, n, D)  same dtype as x

Ascend-oriented optimizations
-----------------------------
1. 2D grid = (T, cdiv(D, BLOCK_D)) : many programs, latency hiding.
2. Load h_res[t] and h_post[t] once into program registers (n*(n+1) scalars).
   This kills the "GM scalar read in inner loop" bottleneck of the AscendC
   version.
3. Inner accumulator kept in fp32; only cast on store. Avoids repeated
   bf16<->fp32 conversions per iteration.
4. HC_MULT=4 fast path fully unrolls the 4x4 accumulation with 16 FMAs and
   keeps all rows live in registers, mirroring the arch35 "R<VL" fastpath
   but expressed portably in Triton.
5. Ascend autotune: num_stages<=2 (software pipeline shallow), moderate
   BLOCK_D between 128 and 1024.

Fallback: generic kernel handles arbitrary n (small).
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HC_MULT=4 fast path
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 1024}, num_warps=8, num_stages=1),
    ],
    key=["D"],
)
@triton.jit
def _mhc_post_kernel_hc4(
    x_ptr,        # (T, 4, D) bf16/fp16
    h_res_ptr,    # (T, 4, 4) fp32   layout: h_res[t, j, i]
    h_out_ptr,    # (T, D)    bf16/fp16
    h_post_ptr,   # (T, 4)    fp32
    out_ptr,      # (T, 4, D) bf16/fp16
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    # ---- base pointers ----
    x_base = pid_t * 4 * D
    out_base = pid_t * 4 * D
    hout_base = pid_t * D
    hres_base = pid_t * 16
    hpost_base = pid_t * 4

    # ---- load per-token scalars once (4 + 16) ----
    hp0 = tl.load(h_post_ptr + hpost_base + 0)
    hp1 = tl.load(h_post_ptr + hpost_base + 1)
    hp2 = tl.load(h_post_ptr + hpost_base + 2)
    hp3 = tl.load(h_post_ptr + hpost_base + 3)

    # h_res[t, j, i] -- laid out row-major as (j, i)
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

    # ---- load x rows and h_out slice, cast to fp32 ----
    x0 = tl.load(x_ptr + x_base + 0 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + x_base + 1 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + x_base + 2 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x3 = tl.load(x_ptr + x_base + 3 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    ho = tl.load(h_out_ptr + hout_base + d_off, mask=d_mask, other=0.0).to(tl.float32)

    # ---- 4 outputs computed simultaneously (independent accumulators) ----
    # out[i] = h_post[i] * ho + sum_j h_res[j,i] * x[j]
    y0 = hp0 * ho + r00 * x0 + r10 * x1 + r20 * x2 + r30 * x3
    y1 = hp1 * ho + r01 * x0 + r11 * x1 + r21 * x2 + r31 * x3
    y2 = hp2 * ho + r02 * x0 + r12 * x1 + r22 * x2 + r32 * x3
    y3 = hp3 * ho + r03 * x0 + r13 * x1 + r23 * x2 + r33 * x3

    dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + out_base + 0 * D + d_off, y0.to(dtype), mask=d_mask)
    tl.store(out_ptr + out_base + 1 * D + d_off, y1.to(dtype), mask=d_mask)
    tl.store(out_ptr + out_base + 2 * D + d_off, y2.to(dtype), mask=d_mask)
    tl.store(out_ptr + out_base + 3 * D + d_off, y3.to(dtype), mask=d_mask)


# ---------------------------------------------------------------------------
# Generic kernel (arbitrary small n)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 512}, num_warps=8, num_stages=1),
    ],
    key=["D", "N"],
)
@triton.jit
def _mhc_post_kernel_generic(
    x_ptr,        # (T, N, D)
    h_res_ptr,    # (T, N, N) fp32
    h_out_ptr,    # (T, D)
    h_post_ptr,   # (T, N)    fp32
    out_ptr,      # (T, N, D)
    D: tl.constexpr,
    N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_i = tl.program_id(1)   # output head
    pid_d = tl.program_id(2)

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    ho = tl.load(h_out_ptr + pid_t * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    hp = tl.load(h_post_ptr + pid_t * N + pid_i).to(tl.float32)
    acc = hp * ho

    x_base = pid_t * N * D
    hres_base = pid_t * N * N
    for j in range(N):
        rj = tl.load(h_res_ptr + hres_base + j * N + pid_i).to(tl.float32)
        xj = tl.load(x_ptr + x_base + j * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
        acc += rj * xj

    dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + pid_t * N * D + pid_i * D + d_off, acc.to(dtype), mask=d_mask)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _flatten_bsn(x, h_res, h_out, h_post):
    """Return (x, h_res, h_out, h_post, restore_shape) in (T, n, D) layout."""
    if x.dim() == 4:
        B, S, N, D = x.shape
        T = B * S
        x = x.reshape(T, N, D).contiguous()
        h_res = h_res.reshape(T, N, N).contiguous()
        h_out = h_out.reshape(T, D).contiguous()
        h_post = h_post.reshape(T, N).contiguous()
        return x, h_res, h_out, h_post, (B, S, N, D)
    if x.dim() == 3:
        T, N, D = x.shape
        return (
            x.contiguous(),
            h_res.reshape(T, N, N).contiguous(),
            h_out.reshape(T, D).contiguous(),
            h_post.reshape(T, N).contiguous(),
            (T, N, D),
        )
    raise ValueError(f"unsupported x.dim()={x.dim()}, expect 3 or 4")


def mhc_post(
    x: torch.Tensor,
    h_res: torch.Tensor,
    h_out: torch.Tensor,
    h_post: torch.Tensor,
) -> torch.Tensor:
    """Fused mhc_post forward. See module docstring for semantics."""
    xf, hres, hout, hpost, shape = _flatten_bsn(x, h_res, h_out, h_post)
    T, N, D = xf.shape
    out = torch.empty_like(xf)

    if N == 4:
        grid = lambda META: (T, triton.cdiv(D, META["BLOCK_D"]))
        _mhc_post_kernel_hc4[grid](xf, hres, hout, hpost, out, D=D)
    else:
        grid = lambda META: (T, N, triton.cdiv(D, META["BLOCK_D"]))
        _mhc_post_kernel_generic[grid](xf, hres, hout, hpost, out, D=D, N=N)

    return out.reshape(shape)


def mhc_post_ref(
    x: torch.Tensor,
    h_res: torch.Tensor,
    h_out: torch.Tensor,
    h_post: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference.

    output[b,s,i,d] = sum_j h_res[b,s,j,i] * x[b,s,j,d]
                    + h_post[b,s,i] * h_out[b,s,d]
    """
    orig_dtype = x.dtype
    xf, hres, hout, hpost, shape = _flatten_bsn(x, h_res, h_out, h_post)
    x_f = xf.float()
    hout_f = hout.float()
    # (T, N, D) = einsum('tji,tjd->tid', hres, x_f)
    mix = torch.einsum("tji,tjd->tid", hres, x_f)
    outer = hpost.unsqueeze(-1) * hout_f.unsqueeze(1)  # (T, N, D)
    y = (mix + outer).to(orig_dtype)
    return y.reshape(shape)

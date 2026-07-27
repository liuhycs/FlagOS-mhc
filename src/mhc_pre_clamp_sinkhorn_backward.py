"""FlagGems-style Triton backward for mhc_pre_clamp_sinkhorn.

Design (hybrid Triton + torch.autograd)
---------------------------------------
The backward of the four-stage forward is naturally split into "big"
(D-dimensional / (T,hcMult,D) sized) and "small" ((T, 24)/(T,4,4) sized)
work. We put the D-sized work in Triton kernels for Ascend efficiency,
and let torch.autograd handle the small Sinkhorn/softmax/clamp/head-affine
backward because:

  * the small stage is a tiny fp32 tensor per token (T * 24 elements),
    so backward-through-autograd is negligible vs the D-sized work;
  * autograd removes the risk of hand-rolled Sinkhorn VJP bugs;
  * the forward already had to save `h_res_logits`, `mixes`, `pre` — reusing
    those under a `torch.enable_grad` re-invocation gives an exact
    numerical match with the FlagGems forward at zero extra memory cost.

Stages and where they run:

  Y-backward (Triton `_y_bwd_kernel_hc4`)
      grad_x_from_y[i,d] = grad_y[i,d] * pre[i]
      grad_pre_from_y[i] = sum_d grad_y[i,d] * x_orig[i,d]

  Small stage backward (torch.autograd)
      inputs   : mixes (T,24), alpha (3,), base (24,), h_res_logits (T,4,4)
      outputs  : post_out (T,4), comb_frag (T,4,4), pre (T,4)
      recompute from `mixes` (fp32), get vjp against grad_post_out /
      grad_comb_frag / grad_pre_from_y -> produces grad_mixes (T,24),
      grad_alpha (3,), grad_base (24,).

  GEMM backward (torch.mm)
      grad_x_scaled = grad_mixes @ phi
      grad_phi      = grad_mixes.T @ x_scaled

  RMSNorm backward (Triton `_rms_bwd_kernel`)
      Given grad_x_scaled (T,HC_D) fp32, inv_rms (T,) fp32,
            x_orig (T,HC_D) input dtype:
      grad_x_from_rms = inv_rms * (grad_x_scaled
                        - (1/HC_D) * inv_rms^2 * x_orig * dot(x_orig, grad_x_scaled))

  Combine:
      grad_x = grad_x_from_y + grad_x_from_rms   (in input dtype)
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

try:
    import triton.experimental.tle.language as tle
    HAS_TLE = True
except ImportError:
    HAS_TLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kernel: hin-backward (modified for reduced hin)
#   grad_hin is (T, D) — broadcast across N=4 heads
#   grad_x_from_hin[t, i, d] = grad_hin[t, d] * pre[t, i]
#   grad_pre_from_hin[t, i]  = sum_d grad_hin[t, d] * x[t, i, d]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 2048}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_D": 4096}, num_warps=4, num_stages=2),
    ],
    key=["D"],
)
@triton.heuristics({"EVEN_D": lambda args: args["D"] % args["BLOCK_D"] == 0})
@triton.jit
def _y_bwd_kernel_hc4(
    grad_y_ptr,         # (T, D)    — same dtype as x (bf16/fp16)
    x_ptr,                # (T, 4, D) — input dtype
    pre_ptr,              # (T, 4)    — fp32  (raw sigmoid, NO eps)
    grad_x_from_y_ptr,  # (T, 4, D) fp32  OUTPUT
    grad_pre_from_y_ptr,  # (T, 4) fp32  OUTPUT
    # strides (element counts, not bytes)
    stride_gy_t,          # grad_y stride along T
    stride_x_t,          # x stride along T  (= 4*D if contiguous)
    stride_x_n,          # x stride along N  (= D if contiguous)
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    pid_t = tl.program_id(0)

    # Stride-based addressing: supports non-contiguous x and grad_y
    gy_base = pid_t * stride_gy_t
    x_row = pid_t * stride_x_t
    ox_base = pid_t * 4 * D  # output is always contiguous (we allocate it)

    p0 = tl.load(pre_ptr + pid_t * 4 + 0)
    p1 = tl.load(pre_ptr + pid_t * 4 + 1)
    p2 = tl.load(pre_ptr + pid_t * 4 + 2)
    p3 = tl.load(pre_ptr + pid_t * 4 + 3)

    acc0 = tl.zeros([], dtype=tl.float32)
    acc1 = tl.zeros([], dtype=tl.float32)
    acc2 = tl.zeros([], dtype=tl.float32)
    acc3 = tl.zeros([], dtype=tl.float32)

    n_full = D // BLOCK_D

    # --- Reverse order: tail block first so d_mask reg dies before main loop ---
    if not EVEN_D:
        d_tail_start = n_full * BLOCK_D
        d_off = d_tail_start + tl.arange(0, BLOCK_D)
        d_mask = d_off < D

        gy = tl.load(grad_y_ptr + gy_base + d_off, mask=d_mask, other=0.0).to(tl.float32)
        x0 = tl.load(x_ptr + x_row + 0 * stride_x_n + d_off, mask=d_mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + x_row + 1 * stride_x_n + d_off, mask=d_mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + x_row + 2 * stride_x_n + d_off, mask=d_mask, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + x_row + 3 * stride_x_n + d_off, mask=d_mask, other=0.0).to(tl.float32)

        tl.store(grad_x_from_y_ptr + ox_base + 0 * D + d_off, gy * p0, mask=d_mask)
        tl.store(grad_x_from_y_ptr + ox_base + 1 * D + d_off, gy * p1, mask=d_mask)
        tl.store(grad_x_from_y_ptr + ox_base + 2 * D + d_off, gy * p2, mask=d_mask)
        tl.store(grad_x_from_y_ptr + ox_base + 3 * D + d_off, gy * p3, mask=d_mask)

        acc0 += tl.sum(gy * x0, axis=0)
        acc1 += tl.sum(gy * x1, axis=0)
        acc2 += tl.sum(gy * x2, axis=0)
        acc3 += tl.sum(gy * x3, axis=0)

    # --- Full blocks: no mask, no mask register alive ---
    for d_start in range(0, n_full * BLOCK_D, BLOCK_D):
        d_off = d_start + tl.arange(0, BLOCK_D)

        gy = tl.load(grad_y_ptr + gy_base + d_off).to(tl.float32)
        x0 = tl.load(x_ptr + x_row + 0 * stride_x_n + d_off).to(tl.float32)
        x1 = tl.load(x_ptr + x_row + 1 * stride_x_n + d_off).to(tl.float32)
        x2 = tl.load(x_ptr + x_row + 2 * stride_x_n + d_off).to(tl.float32)
        x3 = tl.load(x_ptr + x_row + 3 * stride_x_n + d_off).to(tl.float32)

        tl.store(grad_x_from_y_ptr + ox_base + 0 * D + d_off, gy * p0)
        tl.store(grad_x_from_y_ptr + ox_base + 1 * D + d_off, gy * p1)
        tl.store(grad_x_from_y_ptr + ox_base + 2 * D + d_off, gy * p2)
        tl.store(grad_x_from_y_ptr + ox_base + 3 * D + d_off, gy * p3)

        acc0 += tl.sum(gy * x0, axis=0)
        acc1 += tl.sum(gy * x1, axis=0)
        acc2 += tl.sum(gy * x2, axis=0)
        acc3 += tl.sum(gy * x3, axis=0)

    tl.store(grad_pre_from_y_ptr + pid_t * 4 + 0, acc0)
    tl.store(grad_pre_from_y_ptr + pid_t * 4 + 1, acc1)
    tl.store(grad_pre_from_y_ptr + pid_t * 4 + 2, acc2)
    tl.store(grad_pre_from_y_ptr + pid_t * 4 + 3, acc3)


# ---------------------------------------------------------------------------
# Fused kernel: y-backward + rms-backward + combine
#   Eliminates the ~80MB intermediate grad_x_from_y write+read.
#   One program per token. Two passes over HC_D:
#     Pass 1: accumulate dot(x_orig, grad_x_scaled) and grad_pre[n]
#     Pass 2: compute final grad_x = grad_x_from_y + inv*grad_x_scaled + scale*x_orig
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 4096}, num_warps=32, num_stages=2),
    ],
    key=["D"],
)
@triton.jit
def _fused_ybwd_rmsbwd_kernel(
    grad_y_ptr,         # (T, D) input dtype  -- grad of hin
    x_orig_ptr,         # (T, 4, D) input dtype  -- original x (may be non-contiguous)
    grad_x_scaled_ptr,  # (T, 4*D) fp32  -- grad from GEMM bwd (always contiguous)
    inv_rms_ptr,        # (T,) fp32
    pre_ptr,            # (T, 4) fp32
    grad_x_ptr,         # (T, 4*D) input dtype  OUTPUT (always contiguous)
    grad_pre_ptr,       # (T, 4) fp32  OUTPUT
    stride_gy_t,        # grad_y stride along T (elements) — supports non-contiguous
    stride_x_t,         # x stride along T (elements)
    stride_x_n,         # x stride along N (elements)
    D: tl.constexpr,
    HC_D: tl.constexpr,
    D_INV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused y-backward + RMS-backward kernel with TLE async loads.

    Iterates over 4 heads x D-blocks explicitly to avoid expensive
    modulo/division/where operations that dominated the previous version.
    Two passes over the data:
      Pass 1: dot = sum(x * grad_x_scaled), grad_pre[n] = sum(gy * x[n])
      Pass 2: grad_x = gy*pre[n] + inv*gs + scale*x

    Input x and grad_y are accessed via strides to avoid any .contiguous() copy
    even when the caller passes a (B,S,N,D) or (B,S,D) tensor directly.
    Output grad_x is always freshly-allocated contiguous (T, 4*D).

    TLE async loads are applied to all tensor loads to enable prefetching.
    BLOCK_D is autotuned for optimal performance.
    """
    pid = tl.program_id(0)

    # Scalar loads with async hint (TLE required)
    inv = tle.load(inv_rms_ptr + pid, is_async=True)

    ix_base  = pid * stride_x_t    # input x base: stride-based (non-contiguous ok)
    ox_base  = pid * 4 * D          # output grad_x base: always contiguous
    base_gs  = pid * HC_D           # grad_x_scaled base: always contiguous
    igy_base = pid * stride_gy_t   # input grad_y base: stride-based

    # Load pre values once with async hint
    p0 = tle.load(pre_ptr + pid * 4 + 0, is_async=True)
    p1 = tle.load(pre_ptr + pid * 4 + 1, is_async=True)
    p2 = tle.load(pre_ptr + pid * 4 + 2, is_async=True)
    p3 = tle.load(pre_ptr + pid * 4 + 3, is_async=True)

    # Pass 1: accumulate dot and grad_pre
    dot = 0.0
    gp0 = tl.zeros([], dtype=tl.float32)
    gp1 = tl.zeros([], dtype=tl.float32)
    gp2 = tl.zeros([], dtype=tl.float32)
    gp3 = tl.zeros([], dtype=tl.float32)

    NUM_D_BLOCKS = tl.cdiv(D, BLOCK_D)
    for db in range(NUM_D_BLOCKS):
        d_off = db * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_off < D

        # stride-based load with async hint: supports non-contiguous grad_y and x
        gy = tle.load(grad_y_ptr + igy_base + d_off, mask=d_mask, other=0.0, is_async=True).to(tl.float32)

        x0 = tle.load(x_orig_ptr + ix_base + 0 * stride_x_n + d_off, mask=d_mask, other=0.0, is_async=True).to(tl.float32)
        gs0 = tle.load(grad_x_scaled_ptr + base_gs + 0 * D + d_off, mask=d_mask, other=0.0, is_async=True)
        dot += tl.sum(x0 * gs0, axis=0)
        gp0 += tl.sum(gy * x0, axis=0)

        x1 = tle.load(x_orig_ptr + ix_base + 1 * stride_x_n + d_off, mask=d_mask, other=0.0, is_async=True).to(tl.float32)
        gs1 = tle.load(grad_x_scaled_ptr + base_gs + 1 * D + d_off, mask=d_mask, other=0.0, is_async=True)
        dot += tl.sum(x1 * gs1, axis=0)
        gp1 += tl.sum(gy * x1, axis=0)

        x2 = tle.load(x_orig_ptr + ix_base + 2 * stride_x_n + d_off, mask=d_mask, other=0.0, is_async=True).to(tl.float32)
        gs2 = tle.load(grad_x_scaled_ptr + base_gs + 2 * D + d_off, mask=d_mask, other=0.0, is_async=True)
        dot += tl.sum(x2 * gs2, axis=0)
        gp2 += tl.sum(gy * x2, axis=0)

        x3 = tle.load(x_orig_ptr + ix_base + 3 * stride_x_n + d_off, mask=d_mask, other=0.0, is_async=True).to(tl.float32)
        gs3 = tle.load(grad_x_scaled_ptr + base_gs + 3 * D + d_off, mask=d_mask, other=0.0, is_async=True)
        dot += tl.sum(x3 * gs3, axis=0)
        gp3 += tl.sum(gy * x3, axis=0)

    # Store grad_pre
    tl.store(grad_pre_ptr + pid * 4 + 0, gp0)
    tl.store(grad_pre_ptr + pid * 4 + 1, gp1)
    tl.store(grad_pre_ptr + pid * 4 + 2, gp2)
    tl.store(grad_pre_ptr + pid * 4 + 3, gp3)

    # Compute scale for RMS backward
    scale = -D_INV * inv * inv * inv * dot

    # Pass 2: compute final grad_x (output is always contiguous → ox_base)
    dt = grad_x_ptr.dtype.element_ty
    for db in range(NUM_D_BLOCKS):
        d_off = db * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_off < D

        gy = tle.load(grad_y_ptr + igy_base + d_off, mask=d_mask, other=0.0, is_async=True).to(tl.float32)

        x0 = tle.load(x_orig_ptr + ix_base + 0 * stride_x_n + d_off, mask=d_mask, other=0.0, is_async=True).to(tl.float32)
        gs0 = tle.load(grad_x_scaled_ptr + base_gs + 0 * D + d_off, mask=d_mask, other=0.0, is_async=True)
        gx0 = gy * p0 + inv * gs0 + scale * x0
        tl.store(grad_x_ptr + ox_base + 0 * D + d_off, gx0.to(dt), mask=d_mask)

        x1 = tle.load(x_orig_ptr + ix_base + 1 * stride_x_n + d_off, mask=d_mask, other=0.0, is_async=True).to(tl.float32)
        gs1 = tle.load(grad_x_scaled_ptr + base_gs + 1 * D + d_off, mask=d_mask, other=0.0, is_async=True)
        gx1 = gy * p1 + inv * gs1 + scale * x1
        tl.store(grad_x_ptr + ox_base + 1 * D + d_off, gx1.to(dt), mask=d_mask)

        x2 = tle.load(x_orig_ptr + ix_base + 2 * stride_x_n + d_off, mask=d_mask, other=0.0, is_async=True).to(tl.float32)
        gs2 = tle.load(grad_x_scaled_ptr + base_gs + 2 * D + d_off, mask=d_mask, other=0.0, is_async=True)
        gx2 = gy * p2 + inv * gs2 + scale * x2
        tl.store(grad_x_ptr + ox_base + 2 * D + d_off, gx2.to(dt), mask=d_mask)

        x3 = tle.load(x_orig_ptr + ix_base + 3 * stride_x_n + d_off, mask=d_mask, other=0.0, is_async=True).to(tl.float32)
        gs3 = tle.load(grad_x_scaled_ptr + base_gs + 3 * D + d_off, mask=d_mask, other=0.0, is_async=True)
        gx3 = gy * p3 + inv * gs3 + scale * x3
        tl.store(grad_x_ptr + ox_base + 3 * D + d_off, gx3.to(dt), mask=d_mask)


# ---------------------------------------------------------------------------
# Heuristic-optimized vectorized small-stage backward kernel.
# Following baseline patterns: autotune BLOCK_T, explicit range() loops.
# Uses tle.load(is_async=True) for data prefetching when available.
# Processes BLOCK_T tokens per program for vectorized execution.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 64}, num_warps=32, num_stages=2),
    ],
    key=["T", "ITERS"],
)
@triton.heuristics({"EVEN_T": lambda args: args["T"] % args["BLOCK_T"] == 0})
@triton.jit
def _small_stage_bwd_vec_kernel(
    mixes_ptr,           # (T, 24) fp32
    alpha_ptr,           # (3,)    fp32
    base_ptr,            # (24,)   fp32
    grad_pre_ptr,        # (T, 4)   fp32
    grad_post_ptr,       # (T, 4)   fp32
    grad_comb_ptr,       # (T, 16) fp32  (flattened from T,4,4)
    grad_mixes_ptr,      # (T, 24)  fp32     OUT
    grad_alpha_ptr,      # (3,)     fp32     OUT (atomic add)
    grad_base_ptr,       # (24,)    fp32     OUT (atomic add)
    scratch_ptr,         # (T, ITERS, 16) fp32
    T: tl.constexpr,
    HC_EPS: tl.constexpr,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    APPLY_CLAMP: tl.constexpr,
    ITERS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    EVEN_T: tl.constexpr,
):
    """Heuristic-optimized TLE vectorized small-stage backward. BLOCK_T tokens per program."""
    pid = tl.program_id(0)
    t_off = pid * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    offs_4 = tl.arange(0, 4)  # [4]

    # Load constants with TLE async (scalar loads don't need mask/other)
    a0 = tle.load(alpha_ptr + 0, is_async=True)
    a1 = tle.load(alpha_ptr + 1, is_async=True)
    a2 = tle.load(alpha_ptr + 2, is_async=True)
    base_pre = tle.load(base_ptr + offs_4, is_async=True)         # [4]
    base_post = tle.load(base_ptr + 4 + offs_4, is_async=True)    # [4]

    # TLE async prefetch mixes data (split mask vs no-mask path)
    if EVEN_T:
        mix_pre = tle.load(mixes_ptr + t_off[:, None] * 24 + offs_4[None, :],
                           is_async=True)  # [BT,4]
        mix_post = tle.load(mixes_ptr + t_off[:, None] * 24 + (4 + offs_4)[None, :],
                            is_async=True)  # [BT,4]
        ml0 = tle.load(mixes_ptr + t_off[:, None] * 24 + (8 + offs_4)[None, :],
                       is_async=True)
        ml1 = tle.load(mixes_ptr + t_off[:, None] * 24 + (12 + offs_4)[None, :],
                       is_async=True)
        ml2 = tle.load(mixes_ptr + t_off[:, None] * 24 + (16 + offs_4)[None, :],
                       is_async=True)
        ml3 = tle.load(mixes_ptr + t_off[:, None] * 24 + (20 + offs_4)[None, :],
                       is_async=True)
    else:
        t_mask = t_off < T
        mix_pre = tle.load(mixes_ptr + t_off[:, None] * 24 + offs_4[None, :],
                           mask=t_mask[:, None], other=0.0, is_async=True)
        mix_post = tle.load(mixes_ptr + t_off[:, None] * 24 + (4 + offs_4)[None, :],
                            mask=t_mask[:, None], other=0.0, is_async=True)
        ml0 = tle.load(mixes_ptr + t_off[:, None] * 24 + (8 + offs_4)[None, :],
                       mask=t_mask[:, None], other=0.0, is_async=True)
        ml1 = tle.load(mixes_ptr + t_off[:, None] * 24 + (12 + offs_4)[None, :],
                       mask=t_mask[:, None], other=0.0, is_async=True)
        ml2 = tle.load(mixes_ptr + t_off[:, None] * 24 + (16 + offs_4)[None, :],
                       mask=t_mask[:, None], other=0.0, is_async=True)
        ml3 = tle.load(mixes_ptr + t_off[:, None] * 24 + (20 + offs_4)[None, :],
                       mask=t_mask[:, None], other=0.0, is_async=True)

    base_l0 = tle.load(base_ptr + 8 + offs_4, is_async=True)    # [4]
    base_l1 = tle.load(base_ptr + 12 + offs_4, is_async=True)   # [4]
    base_l2 = tle.load(base_ptr + 16 + offs_4, is_async=True)   # [4]
    base_l3 = tle.load(base_ptr + 20 + offs_4, is_async=True)   # [4]

    # ======== FORWARD RECOMPUTE (vectorized) ========
    z_pre = mix_pre * a0 + base_pre[None, :]
    sp = tl.sigmoid(z_pre)  # [BT, 4]
    z_post = mix_post * a1 + base_post[None, :]
    spo = tl.sigmoid(z_post)  # [BT, 4]

    # Logits per row + clamp
    l0 = ml0 * a2 + base_l0[None, :]  # [BT, 4] row 0
    l1 = ml1 * a2 + base_l1[None, :]  # [BT, 4] row 1
    l2 = ml2 * a2 + base_l2[None, :]  # [BT, 4] row 2
    l3 = ml3 * a2 + base_l3[None, :]  # [BT, 4] row 3
    if APPLY_CLAMP:
        c0 = tl.minimum(tl.maximum(l0, CLAMP_MIN), CLAMP_MAX)
        c1 = tl.minimum(tl.maximum(l1, CLAMP_MIN), CLAMP_MAX)
        c2 = tl.minimum(tl.maximum(l2, CLAMP_MIN), CLAMP_MAX)
        c3 = tl.minimum(tl.maximum(l3, CLAMP_MIN), CLAMP_MAX)
    else:
        c0 = l0; c1 = l1; c2 = l2; c3 = l3

    # Row-wise softmax
    mx0 = tl.max(c0, axis=1)[:, None]; mx1 = tl.max(c1, axis=1)[:, None]
    mx2 = tl.max(c2, axis=1)[:, None]; mx3 = tl.max(c3, axis=1)[:, None]
    e0 = tl.exp(c0 - mx0); e1 = tl.exp(c1 - mx1)
    e2 = tl.exp(c2 - mx2); e3 = tl.exp(c3 - mx3)
    rs0 = tl.sum(e0, axis=1)[:, None]; rs1 = tl.sum(e1, axis=1)[:, None]
    rs2 = tl.sum(e2, axis=1)[:, None]; rs3 = tl.sum(e3, axis=1)[:, None]
    v0 = e0 / rs0 + HC_EPS; v1 = e1 / rs1 + HC_EPS
    v2 = e2 / rs2 + HC_EPS; v3 = e3 / rs3 + HC_EPS

    # Iter 0: col-norm
    cs = v0 + v1 + v2 + v3 + HC_EPS
    v0 = v0 / cs; v1 = v1 / cs; v2 = v2 / cs; v3 = v3 / cs

    # Store iter 0 to scratch (vectorized: 4 stores of [BT,4] vs 16 scalar stores)
    scr_stride: tl.constexpr = ITERS * 16
    scr_base = t_off * scr_stride  # [BT] base offset per token
    if EVEN_T:
        tl.store(scratch_ptr + scr_base[:, None] + offs_4[None, :], v0)
        tl.store(scratch_ptr + scr_base[:, None] + (4 + offs_4)[None, :], v1)
        tl.store(scratch_ptr + scr_base[:, None] + (8 + offs_4)[None, :], v2)
        tl.store(scratch_ptr + scr_base[:, None] + (12 + offs_4)[None, :], v3)
    else:
        tl.store(scratch_ptr + scr_base[:, None] + offs_4[None, :], v0, mask=t_mask[:, None])
        tl.store(scratch_ptr + scr_base[:, None] + (4 + offs_4)[None, :], v1, mask=t_mask[:, None])
        tl.store(scratch_ptr + scr_base[:, None] + (8 + offs_4)[None, :], v2, mask=t_mask[:, None])
        tl.store(scratch_ptr + scr_base[:, None] + (12 + offs_4)[None, :], v3, mask=t_mask[:, None])

    # Iters 1..ITERS-1 (explicit range() for compiler optimization)
    for k in range(1, ITERS):
        r0s = tl.sum(v0, axis=1)[:, None] + HC_EPS
        r1s = tl.sum(v1, axis=1)[:, None] + HC_EPS
        r2s = tl.sum(v2, axis=1)[:, None] + HC_EPS
        r3s = tl.sum(v3, axis=1)[:, None] + HC_EPS
        v0 = v0 / r0s; v1 = v1 / r1s; v2 = v2 / r2s; v3 = v3 / r3s
        cs = v0 + v1 + v2 + v3 + HC_EPS
        v0 = v0 / cs; v1 = v1 / cs; v2 = v2 / cs; v3 = v3 / cs
        k_off = k * 16
        if EVEN_T:
            tl.store(scratch_ptr + scr_base[:, None] + (k_off + offs_4)[None, :], v0)
            tl.store(scratch_ptr + scr_base[:, None] + (k_off + 4 + offs_4)[None, :], v1)
            tl.store(scratch_ptr + scr_base[:, None] + (k_off + 8 + offs_4)[None, :], v2)
            tl.store(scratch_ptr + scr_base[:, None] + (k_off + 12 + offs_4)[None, :], v3)
        else:
            tl.store(scratch_ptr + scr_base[:, None] + (k_off + offs_4)[None, :], v0, mask=t_mask[:, None])
            tl.store(scratch_ptr + scr_base[:, None] + (k_off + 4 + offs_4)[None, :], v1, mask=t_mask[:, None])
            tl.store(scratch_ptr + scr_base[:, None] + (k_off + 8 + offs_4)[None, :], v2, mask=t_mask[:, None])
            tl.store(scratch_ptr + scr_base[:, None] + (k_off + 12 + offs_4)[None, :], v3, mask=t_mask[:, None])

    # ======== BACKWARD ========
    if EVEN_T:
        g0 = tle.load(grad_comb_ptr + t_off[:, None] * 16 + offs_4[None, :], is_async=True)
        g1 = tle.load(grad_comb_ptr + t_off[:, None] * 16 + (4 + offs_4)[None, :], is_async=True)
        g2 = tle.load(grad_comb_ptr + t_off[:, None] * 16 + (8 + offs_4)[None, :], is_async=True)
        g3 = tle.load(grad_comb_ptr + t_off[:, None] * 16 + (12 + offs_4)[None, :], is_async=True)
    else:
        g0 = tle.load(grad_comb_ptr + t_off[:, None] * 16 + offs_4[None, :],
                      mask=t_mask[:, None], other=0.0, is_async=True)
        g1 = tle.load(grad_comb_ptr + t_off[:, None] * 16 + (4 + offs_4)[None, :],
                      mask=t_mask[:, None], other=0.0, is_async=True)
        g2 = tle.load(grad_comb_ptr + t_off[:, None] * 16 + (8 + offs_4)[None, :],
                      mask=t_mask[:, None], other=0.0, is_async=True)
        g3 = tle.load(grad_comb_ptr + t_off[:, None] * 16 + (12 + offs_4)[None, :],
                      mask=t_mask[:, None], other=0.0, is_async=True)

    # Walk iters backward: ITERS-1 .. 1 (explicit range())
    for k_rev in range(1, ITERS):
        k = ITERS - k_rev
        k_off = k * 16
        pk_off = (k - 1) * 16
        # Load u (iter k state) and p (iter k-1 state) from scratch with TLE async
        if EVEN_T:
            u0 = tle.load(scratch_ptr + scr_base[:, None] + (k_off + offs_4)[None, :], is_async=True)
            u1 = tle.load(scratch_ptr + scr_base[:, None] + (k_off + 4 + offs_4)[None, :], is_async=True)
            u2 = tle.load(scratch_ptr + scr_base[:, None] + (k_off + 8 + offs_4)[None, :], is_async=True)
            u3 = tle.load(scratch_ptr + scr_base[:, None] + (k_off + 12 + offs_4)[None, :], is_async=True)
            p0 = tle.load(scratch_ptr + scr_base[:, None] + (pk_off + offs_4)[None, :], is_async=True)
            p1 = tle.load(scratch_ptr + scr_base[:, None] + (pk_off + 4 + offs_4)[None, :], is_async=True)
            p2 = tle.load(scratch_ptr + scr_base[:, None] + (pk_off + 8 + offs_4)[None, :], is_async=True)
            p3 = tle.load(scratch_ptr + scr_base[:, None] + (pk_off + 12 + offs_4)[None, :], is_async=True)
        else:
            u0 = tle.load(scratch_ptr + scr_base[:, None] + (k_off + offs_4)[None, :], mask=t_mask[:, None], other=0.0, is_async=True)
            u1 = tle.load(scratch_ptr + scr_base[:, None] + (k_off + 4 + offs_4)[None, :], mask=t_mask[:, None], other=0.0, is_async=True)
            u2 = tle.load(scratch_ptr + scr_base[:, None] + (k_off + 8 + offs_4)[None, :], mask=t_mask[:, None], other=0.0, is_async=True)
            u3 = tle.load(scratch_ptr + scr_base[:, None] + (k_off + 12 + offs_4)[None, :], mask=t_mask[:, None], other=0.0, is_async=True)
            p0 = tle.load(scratch_ptr + scr_base[:, None] + (pk_off + offs_4)[None, :], mask=t_mask[:, None], other=0.0, is_async=True)
            p1 = tle.load(scratch_ptr + scr_base[:, None] + (pk_off + 4 + offs_4)[None, :], mask=t_mask[:, None], other=0.0, is_async=True)
            p2 = tle.load(scratch_ptr + scr_base[:, None] + (pk_off + 8 + offs_4)[None, :], mask=t_mask[:, None], other=0.0, is_async=True)
            p3 = tle.load(scratch_ptr + scr_base[:, None] + (pk_off + 12 + offs_4)[None, :], mask=t_mask[:, None], other=0.0, is_async=True)

        # Recompute row/col norms for backward
        Sr0 = tl.sum(p0, axis=1)[:, None]; ir0 = 1.0 / (Sr0 + HC_EPS)
        Sr1 = tl.sum(p1, axis=1)[:, None]; ir1 = 1.0 / (Sr1 + HC_EPS)
        Sr2 = tl.sum(p2, axis=1)[:, None]; ir2 = 1.0 / (Sr2 + HC_EPS)
        Sr3 = tl.sum(p3, axis=1)[:, None]; ir3 = 1.0 / (Sr3 + HC_EPS)
        q0 = p0 * ir0; q1 = p1 * ir1; q2 = p2 * ir2; q3 = p3 * ir3
        Sc = q0 + q1 + q2 + q3; ic = 1.0 / (Sc + HC_EPS)

        # Col-norm backward: dq = ic * (g - u * col_dot)
        sc_dot = g0 * u0 + g1 * u1 + g2 * u2 + g3 * u3
        dq0 = ic * (g0 - sc_dot); dq1 = ic * (g1 - sc_dot)
        dq2 = ic * (g2 - sc_dot); dq3 = ic * (g3 - sc_dot)

        # Row-norm backward: dp = ir * (dq - q * row_dot)
        sr_dot0 = tl.sum(dq0 * q0, axis=1)[:, None]
        sr_dot1 = tl.sum(dq1 * q1, axis=1)[:, None]
        sr_dot2 = tl.sum(dq2 * q2, axis=1)[:, None]
        sr_dot3 = tl.sum(dq3 * q3, axis=1)[:, None]
        g0 = ir0 * (dq0 - sr_dot0); g1 = ir1 * (dq1 - sr_dot1)
        g2 = ir2 * (dq2 - sr_dot2); g3 = ir3 * (dq3 - sr_dot3)

    # Iter 0 backward: col-norm → softmax
    sinv_r0 = 1.0 / rs0; sinv_r1 = 1.0 / rs1
    sinv_r2 = 1.0 / rs2; sinv_r3 = 1.0 / rs3
    sm0b = e0 * sinv_r0 + HC_EPS; sm1b = e1 * sinv_r1 + HC_EPS
    sm2b = e2 * sinv_r2 + HC_EPS; sm3b = e3 * sinv_r3 + HC_EPS
    Sc0 = sm0b + sm1b + sm2b + sm3b + HC_EPS; ic0 = 1.0 / Sc0
    u0b = sm0b * ic0; u1b = sm1b * ic0; u2b = sm2b * ic0; u3b = sm3b * ic0

    sc_dot_0 = g0 * u0b + g1 * u1b + g2 * u2b + g3 * u3b
    dsm0 = ic0 * (g0 - sc_dot_0); dsm1 = ic0 * (g1 - sc_dot_0)
    dsm2 = ic0 * (g2 - sc_dot_0); dsm3 = ic0 * (g3 - sc_dot_0)

    # Softmax backward
    s0v = e0 * sinv_r0; s1v = e1 * sinv_r1; s2v = e2 * sinv_r2; s3v = e3 * sinv_r3
    sd0 = tl.sum(dsm0 * s0v, axis=1)[:, None]
    sd1 = tl.sum(dsm1 * s1v, axis=1)[:, None]
    sd2 = tl.sum(dsm2 * s2v, axis=1)[:, None]
    sd3 = tl.sum(dsm3 * s3v, axis=1)[:, None]
    dc0 = s0v * (dsm0 - sd0); dc1 = s1v * (dsm1 - sd1)
    dc2 = s2v * (dsm2 - sd2); dc3 = s3v * (dsm3 - sd3)

    # Clamp backward
    if APPLY_CLAMP:
        dc0 = tl.where((l0 > CLAMP_MIN) & (l0 < CLAMP_MAX), dc0, 0.0)
        dc1 = tl.where((l1 > CLAMP_MIN) & (l1 < CLAMP_MAX), dc1, 0.0)
        dc2 = tl.where((l2 > CLAMP_MIN) & (l2 < CLAMP_MAX), dc2, 0.0)
        dc3 = tl.where((l3 > CLAMP_MIN) & (l3 < CLAMP_MAX), dc3, 0.0)

    # grad_alpha partials
    da2 = tl.sum(dc0 * ml0, axis=1) + tl.sum(dc1 * ml1, axis=1) \
        + tl.sum(dc2 * ml2, axis=1) + tl.sum(dc3 * ml3, axis=1)
    if EVEN_T:
        grad_pre_v = tle.load(grad_pre_ptr + t_off[:, None] * 4 + offs_4[None, :], is_async=True)
    else:
        grad_pre_v = tle.load(grad_pre_ptr + t_off[:, None] * 4 + offs_4[None, :],
                              mask=t_mask[:, None], other=0.0, is_async=True)
    dz_pre = grad_pre_v * sp * (1.0 - sp)
    da0 = tl.sum(dz_pre * mix_pre, axis=1)
    if EVEN_T:
        grad_post_v = tle.load(grad_post_ptr + t_off[:, None] * 4 + offs_4[None, :], is_async=True)
    else:
        grad_post_v = tle.load(grad_post_ptr + t_off[:, None] * 4 + offs_4[None, :],
                               mask=t_mask[:, None], other=0.0, is_async=True)
    dz_post = grad_post_v * 2.0 * spo * (1.0 - spo)
    da1 = tl.sum(dz_post * mix_post, axis=1)

    # Store grad_mixes [BT, 24]
    if EVEN_T:
        tl.store(grad_mixes_ptr + t_off[:, None] * 24 + offs_4[None, :], dz_pre * a0)
        tl.store(grad_mixes_ptr + t_off[:, None] * 24 + (4 + offs_4)[None, :], dz_post * a1)
        tl.store(grad_mixes_ptr + t_off[:, None] * 24 + (8 + offs_4)[None, :], dc0 * a2)
        tl.store(grad_mixes_ptr + t_off[:, None] * 24 + (12 + offs_4)[None, :], dc1 * a2)
        tl.store(grad_mixes_ptr + t_off[:, None] * 24 + (16 + offs_4)[None, :], dc2 * a2)
        tl.store(grad_mixes_ptr + t_off[:, None] * 24 + (20 + offs_4)[None, :], dc3 * a2)
    else:
        tl.store(grad_mixes_ptr + t_off[:, None] * 24 + offs_4[None, :], dz_pre * a0, mask=t_mask[:, None])
        tl.store(grad_mixes_ptr + t_off[:, None] * 24 + (4 + offs_4)[None, :], dz_post * a1, mask=t_mask[:, None])
        tl.store(grad_mixes_ptr + t_off[:, None] * 24 + (8 + offs_4)[None, :], dc0 * a2, mask=t_mask[:, None])
        tl.store(grad_mixes_ptr + t_off[:, None] * 24 + (12 + offs_4)[None, :], dc1 * a2, mask=t_mask[:, None])
        tl.store(grad_mixes_ptr + t_off[:, None] * 24 + (16 + offs_4)[None, :], dc2 * a2, mask=t_mask[:, None])
        tl.store(grad_mixes_ptr + t_off[:, None] * 24 + (20 + offs_4)[None, :], dc3 * a2, mask=t_mask[:, None])

    # Atomic add to grad_alpha [3] and grad_base [24]
    # Sum across BLOCK_T tokens before atomic add to reduce contention
    da0_sum = tl.sum(da0)
    da1_sum = tl.sum(da1)
    da2_sum = tl.sum(da2)
    tl.atomic_add(grad_alpha_ptr + 0, da0_sum)
    tl.atomic_add(grad_alpha_ptr + 1, da1_sum)
    tl.atomic_add(grad_alpha_ptr + 2, da2_sum)

    # For grad_base, sum across BLOCK_T tokens for each of 24 elements
    dz_pre_sum = tl.sum(dz_pre, axis=0)  # (4,)
    dz_post_sum = tl.sum(dz_post, axis=0)  # (4,)
    dc0_sum = tl.sum(dc0, axis=0)  # (4,)
    dc1_sum = tl.sum(dc1, axis=0)  # (4,)
    dc2_sum = tl.sum(dc2, axis=0)  # (4,)
    dc3_sum = tl.sum(dc3, axis=0)  # (4,)

    tl.atomic_add(grad_base_ptr + offs_4, dz_pre_sum)
    tl.atomic_add(grad_base_ptr + (4 + offs_4), dz_post_sum)
    tl.atomic_add(grad_base_ptr + (8 + offs_4), dc0_sum)
    tl.atomic_add(grad_base_ptr + (12 + offs_4), dc1_sum)
    tl.atomic_add(grad_base_ptr + (16 + offs_4), dc2_sum)
    tl.atomic_add(grad_base_ptr + (20 + offs_4), dc3_sum)


def _small_stage_bwd(
    mixes: torch.Tensor,          # (T, 24) fp32
    alpha: torch.Tensor,          # (3,)    fp32
    base: torch.Tensor,           # (24,)   fp32
    h_res_logits: torch.Tensor,   # (T, 4, 4) fp32   pre-clamp logits (not strictly needed; recomputed)
    grad_pre: torch.Tensor,       # (T, 4)  fp32
    grad_post_out: torch.Tensor,  # (T, 4)  fp32
    grad_comb_frag: torch.Tensor, # (T, 4, 4) fp32
    hc_eps: float,
    clamp_min: float,
    clamp_max: float,
    iter_times: int,
):
    """Fused Triton small-stage backward with atomic add for grad_alpha/grad_base."""
    T = mixes.shape[0]
    grad_mixes = torch.empty((T, 24), dtype=torch.float32, device=mixes.device)
    grad_alpha = torch.zeros((3,), dtype=torch.float32, device=mixes.device)
    grad_base = torch.zeros((24,), dtype=torch.float32, device=mixes.device)
    scratch = torch.empty((T, iter_times, 16), dtype=torch.float32, device=mixes.device)
    apply_clamp = 1 if (clamp_min != 0.0 or clamp_max != 0.0) else 0

    # Use heuristic-optimized vectorized kernel with autotune
    grid = lambda meta: ((T + meta["BLOCK_T"] - 1) // meta["BLOCK_T"],)

    _small_stage_bwd_vec_kernel[grid](
        mixes, alpha, base,
        grad_pre, grad_post_out, grad_comb_frag.reshape(T, 16),
        grad_mixes, grad_alpha, grad_base, scratch,
        T=T,
        HC_EPS=hc_eps,
        CLAMP_MIN=float(clamp_min),
        CLAMP_MAX=float(clamp_max),
        APPLY_CLAMP=apply_clamp,
        ITERS=int(iter_times),
    )
    return grad_mixes, grad_alpha, grad_base


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def mhc_pre_clamp_sinkhorn_backward(
    # forward saved
    x: torch.Tensor,                 # (B,S,N,D) or (T,N,D) original x
    phi: torch.Tensor,               # (hcMix, hcMult*D) fp32
    alpha: torch.Tensor,             # (3,)
    base: torch.Tensor,              # (hcMix,)
    inv_rms: torch.Tensor,           # (T,) fp32
    x_scaled: torch.Tensor,          # (T, HC_D) fp32
    mixes: torch.Tensor,             # (T, hcMix) fp32
    h_res_logits: torch.Tensor,      # (T, hcMult, hcMult) fp32   (unused; autograd recomputes)
    pre: torch.Tensor,               # (T, hcMult) fp32
    # grad inputs
    grad_y: torch.Tensor,            # (B,S,D) or (T,D) same dtype as x
    grad_post_out: torch.Tensor,     # (T, hcMult) fp32
    grad_comb_frag: torch.Tensor,    # (T, hcMult, hcMult) fp32
    # hyperparams
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    clamp_min: float = 0.0,
    clamp_max: float = 0.0,
    iter_times: int = 20,
):
    """Full backward.  Returns (grad_x, grad_phi, grad_alpha, grad_base).

    Layout-agnostic: accepts x as (B,S,N,D) or (T,N,D), grad_y as (B,S,D)
    or (T,D) without any reshape/contiguous copy.  Shape and strides are
    extracted directly from the original tensors and forwarded to kernels.
    """
    # ---- Extract shape & strides without reshape/flatten ----
    if x.dim() == 4:
        B, S, N, D = x.shape
        T = B * S
        shape = (B, S, N, D)
        # stride along the token (T) dimension: jump one S-row in the B*S view
        stride_x_t = x.stride(1)       # stride(S dim) = N*D when contiguous
        stride_x_n = x.stride(2)       # stride(N dim) = D   when contiguous
    elif x.dim() == 3:
        T, N, D = x.shape
        shape = (T, N, D)
        stride_x_t = x.stride(0)
        stride_x_n = x.stride(1)
    else:
        raise ValueError(f"unsupported x.dim()={x.dim()}")

    if grad_y.dim() == 3:
        _B, _S, _D = grad_y.shape
        assert _B * _S == T and _D == D
        stride_gy_t = grad_y.stride(1)  # stride(S dim) in the B*S view
    elif grad_y.dim() == 2:
        assert grad_y.shape == (T, D)
        stride_gy_t = grad_y.stride(0)
    else:
        raise ValueError(f"unsupported grad_y.dim()={grad_y.dim()}")

    assert N == 4
    hc_d  = N * D
    hc_mix = N * (N + 2)  # noqa: F841

    # ---- Y-backward (only for grad_pre_from_y) ----
    grad_pre_from_y = torch.empty(T, N, dtype=torch.float32, device=x.device)
    grad_x_from_y_tmp = torch.empty(T, N, D, dtype=torch.float32, device=x.device)
    _y_bwd_kernel_hc4[(T,)](
        grad_y, x, pre,
        grad_x_from_y_tmp, grad_pre_from_y,
        stride_gy_t=stride_gy_t,
        stride_x_t=stride_x_t,
        stride_x_n=stride_x_n,
        D=D,
    )

    # ---- Small-stage bwd (fused Triton) ----
    grad_mixes, grad_alpha, grad_base = _small_stage_bwd(
        mixes, alpha.to(torch.float32), base.to(torch.float32),
        h_res_logits,
        grad_pre_from_y, grad_post_out, grad_comb_frag,
        hc_eps, clamp_min, clamp_max, iter_times,
    )

    # ---- GEMM bwd ----
    phi_f = phi.to(torch.float32)
    grad_x_scaled = torch.mm(grad_mixes, phi_f)    # (T, HC_D)
    grad_phi      = torch.mm(grad_mixes.t(), x_scaled)  # (hcMix, HC_D)

    # ---- Fused RMSNorm bwd + Y-backward combine ----
    grad_x = torch.empty(T, hc_d, dtype=x.dtype, device=x.device)
    _fused_ybwd_rmsbwd_kernel[(T,)](
        grad_y, x, grad_x_scaled, inv_rms, pre,
        grad_x, grad_pre_from_y,
        stride_gy_t=stride_gy_t,
        stride_x_t=stride_x_t,
        stride_x_n=stride_x_n,
        D=D, HC_D=hc_d, D_INV=1.0 / hc_d,
    )

    return (
        grad_x.view(*shape),
        grad_phi.reshape(phi.shape),
        grad_alpha,
        grad_base,
    )


# ---------------------------------------------------------------------------
# Reference implementation (autograd through forward)
# ---------------------------------------------------------------------------
def mhc_pre_clamp_sinkhorn_backward_ref(
    x, phi, alpha, base,
    grad_y, grad_post_out, grad_comb_frag,
    norm_eps=1e-6, hc_eps=1e-6,
    clamp_min=0.0, clamp_max=0.0,
    iter_times=20,
):
    """Pure PyTorch reference (autograd through forward). Returns
    (grad_x, grad_phi, grad_alpha, grad_base).
    """
    orig_dtype = x.dtype
    x_ = x.detach().to(torch.float32).requires_grad_(True)
    phi_ = phi.detach().to(torch.float32).requires_grad_(True)
    alpha_ = alpha.detach().to(torch.float32).requires_grad_(True)
    base_ = base.detach().to(torch.float32).requires_grad_(True)

    if x_.dim() == 4:
        B, S, N, D = x_.shape
        xf = x_.reshape(B * S, N, D)
    else:
        xf = x_
        N = xf.shape[-2]
        D = xf.shape[-1]

    T = xf.shape[0]
    x_flat = xf.reshape(T, N * D)
    ms = x_flat.pow(2).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(ms + norm_eps)
    x_scaled = x_flat * inv
    mixes = x_scaled @ phi_.t()

    a = alpha_
    b = base_
    pre = torch.sigmoid(mixes[:, :N] * a[0] + b[:N]) + hc_eps
    # post_out = 2 * sigmoid(...) per aclnn spec.
    post_out = 2.0 * torch.sigmoid(mixes[:, N:2*N] * a[1] + b[N:2*N])
    logits = (mixes[:, 2*N:] * a[2] + b[2*N:]).reshape(T, N, N)
    if clamp_min != 0.0 or clamp_max != 0.0:
        logits_c = torch.clamp(logits, clamp_min, clamp_max)
    else:
        logits_c = logits
    row_max = logits_c.max(dim=-1, keepdim=True).values
    M = (logits_c - row_max).exp()
    M = M / M.sum(dim=-1, keepdim=True) + hc_eps
    M = M / (M.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(iter_times - 1):
        M = M / (M.sum(dim=-1, keepdim=True) + hc_eps)
        M = M / (M.sum(dim=-2, keepdim=True) + hc_eps)
    # hin = sum_n(x * pre) -> (T, D). aclnn semantic.
    y = (xf * pre.unsqueeze(-1)).sum(dim=-2)   # (T, D)

    gy = grad_y.to(torch.float32).reshape(T, D)
    loss = (y * gy).sum() \
         + (post_out * grad_post_out).sum() \
         + (M * grad_comb_frag).sum()
    dx, dphi, da, db = torch.autograd.grad(loss, (x_, phi_, alpha_, base_))
    return dx.to(orig_dtype), dphi, da, db

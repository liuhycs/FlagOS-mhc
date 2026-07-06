# MHC operators — Triton implementation for Ascend NPU

Triton implementations of the four MHC operators. Each operator has:

- a Triton fast path tuned for Ascend NPU (AscendC-style techniques expressed
  portably in Triton)
- a pure-PyTorch reference (`*_ref`) for correctness testing
- a `torch.autograd.Function` wrapper (`mhc_post_fn`,
  `mhc_pre_clamp_sinkhorn_fn`) for the forward+backward pair

## Layout

```
FlagOS-mhc/
├── README.md
├── src/
│   ├── __init__.py                        Public exports
│   ├── autograd.py                        MhcPostFn, MhcPreClampSinkhornFn
│   ├── mhc_post.py                        mhc_post + mhc_post_ref (hc4 fastpath + generic)
│   ├── mhc_post_backward.py               mhc_post_backward + ref
│   ├── mhc_pre_clamp_sinkhorn.py          mhc_pre_clamp_sinkhorn + ref (aclnn semantic)
│   └── mhc_pre_clamp_sinkhorn_backward.py mhc_pre_clamp_sinkhorn_backward + ref
└── tests/
    └── test_correctness.py                Reference-vs-Triton correctness harness
```

## Semantics (matches production AscendC operators)

### `mhc_post`
```
output[b,s,i,d] = sum_j h_res[b,s,j,i] * x[b,s,j,d]
                + h_post[b,s,i] * h_out[b,s,d]
```
Inputs: `x (T,N,D)`, `h_res (T,N,N)`, `h_out (T,D)`, `h_post (T,N)`.
Both `(B,S,N,D)` and `(T,N,D)` inputs are accepted; the kernel flattens
internally with `T = B*S` and reshapes the output back.

### `mhc_post_backward`
Standard VJPs of the above, computed by two Triton kernels
(`_mhc_post_bwd_dxdho_hc4` for `grad_x`/`grad_h_out`,
`_mhc_post_bwd_dresdpost_hc4` for `grad_h_res`/`grad_h_post`):
- `grad_x[j,d]      = sum_i h_res[j,i] * grad_y[i,d]`
- `grad_h_res[j,i]  = sum_d x[j,d] * grad_y[i,d]`
- `grad_h_out[d]    = sum_i h_post[i] * grad_y[i,d]`
- `grad_h_post[i]   = sum_d h_out[d] * grad_y[i,d]`

The reductions in kernel 2 are done with **plain stores** (one program per
token, accumulator lives entirely in fp32 registers); no `tl.atomic_add`
is used.

### `mhc_pre_clamp_sinkhorn`
Five-stage fused pipeline (aclnn semantic):

1. RMSNorm over `(hcMult*D)` axis, produce `inv_rms` and `x_scaled`
2. Heavy GEMM `mixes = x_scaled @ phi.T`  (offloaded to `torch.mm`)
3. Three-head affine split: `pre = σ(mixes[:N]*α[0]+base[:N]) + hc_eps`,
   `post_out = 2·σ(mixes[N:2N]*α[1]+base[N:2N])`,
   `logits = mixes[2N:]*α[2]+base[2N:]`
4. Optional clamp on logits, row-softmax (subtract row-max for stability),
   then 20 iterations of `(M += hc_eps; col-norm; row-norm; col-norm)`
5. `y = sum_n x * pre` (aclnn `hin` output, shape `(B,S,D)` or `(T,D)`)

The function returns a dict; when `need_backward=True` it additionally
saves `inv_rms`, `x_scaled`, `mixes`, `h_res_logits`, `pre`.

### `mhc_pre_clamp_sinkhorn_backward`
Hybrid design: Triton for the D-dimensional work (`_y_bwd_kernel_hc4`,
`_rms_bwd_combine_kernel`) and for the tiny per-token Sinkhorn/softmax/
clamp/head-affine backward (`_small_stage_bwd_kernel_hc4`, which keeps all
16 combFrag values + iter scratch in registers). The two GEMMs
(`grad_x_scaled = grad_mixes @ phi`, `grad_phi = grad_mixes.T @ x_scaled`)
are offloaded to `torch.mm`. The kernel writes each `grad_x` element once
(no atomic on GM).

## Ascend-oriented optimizations

| Bottleneck in AscendC                            | Triton handling here                                                      |
|--------------------------------------------------|---------------------------------------------------------------------------|
| GM scalar reads + `CACHE_MODE_DISABLE`           | `tl.load` once per program, scalars kept in fp32 registers                |
| Queue-depth 1, no double buffer                  | `num_stages=1..2` autotune configs, one program/token                     |
| Manual `PipeBarrier<PIPE_V>` / SetFlag           | Removed — Triton handles synchronization implicitly                      |
| n=4 hard-coded loops, tail-C duplicate           | `hc4` fastpaths fully unroll 4 heads; `_generic` uses a masked loop      |
| `smallBufferBytes` manual bookkeeping           | Compiler handles tiling                                                  |
| `gradOutF32` scanned three times                | Single-pass per kernel; one load of `grad_y`                              |
| `Duplicate` + `Div` broadcast per iter          | Native broadcast + reciprocal-then-multiply                               |
| bf16 ↔ fp32 casts on every op                    | Implicit fp32 accum; cast only on store                                  |
| Sinkhorn UB round-trips (store+reload each iter) | All 20 iterations + scratch live in registers                            |
| `Duplicate → Div` per iteration                  | `inv_s = 1.0/(s+eps)` computed once, then multiplied                      |
| Atomic on GM for reductions                      | One program/token + register accumulators → plain store, no contention   |

Additional Ascend-specific choices:

1. **Heavy GEMMs delegated to `torch.mm`.** Ascend has fast MM through CANN;
   `tl.dot` on Triton-Ascend for `(T, HC_D) x (HC_D, hcMix)` shape would
   need careful ND2NZ handling and typically loses vs the library.
2. **`num_stages` capped at 1..2.** Deep pipelining hits UB pressure fast
   on Ascend AIV; shallow stages keep the software pipeliner happy.
3. **No `tl.atomic_add` in the hot paths.** Reductions in
   `_mhc_post_bwd_dresdpost_hc4`, `_y_bwd_kernel_hc4`,
   `_small_stage_bwd_kernel_hc4` are computed by a single program that
   owns the full per-token accumulator and writes the final values once.
4. **Kernel-level split matches production stages** (RMS, GEMM,
   heads+Sinkhorn, y-scale; on backward: y-bwd, small-stage-bwd,
   GEMM-bwd, rms-bwd+combine). Each kernel is fully register-resident in
   the hcMult=4 fastpath, so no intra-kernel GM spills.
5. **`hc4` fastpaths** fully unroll all `4x4` and `4` state to independent
   scalar accumulators. The compiler emits FMA chains without needing
   manual `MIX_AIC_1_2` orchestration.

## Public API

```python
from FlagOS_mhc import (   # add src/ to sys.path or install as a package
    # forward / backward operators
    mhc_post,                                  # non-autograd
    mhc_post_backward,
    mhc_pre_clamp_sinkhorn,                    # returns dict
    mhc_pre_clamp_sinkhorn_backward,
    # autograd wrappers
    mhc_post_fn,                               # torch.autograd.Function
    mhc_pre_clamp_sinkhorn_fn,                 # returns (y, post_out, comb_frag)
    MhcPostFn,
    MhcPreClampSinkhornFn,
    # pure-PyTorch references
    mhc_post_ref,
    mhc_post_backward_ref,
    mhc_pre_clamp_sinkhorn_ref,
    mhc_pre_clamp_sinkhorn_backward_ref,
)
```

Autograd usage:

```python
import torch
from FlagOS_mhc import mhc_post_fn, mhc_pre_clamp_sinkhorn_fn

# mhc_post
y = mhc_post_fn(x, h_res, h_out, h_post)
y.sum().backward()

# mhc_pre_clamp_sinkhorn
y, post, comb = mhc_pre_clamp_sinkhorn_fn(
    x, phi, alpha, base,
    norm_eps=1e-6, hc_eps=1e-6,
    clamp_min=-3.0, clamp_max=3.0,
    iter_times=20,
)
(y.sum() + comb.sum()).backward()
```

`mhc_pre_clamp_sinkhorn_fn` defaults are `clamp_min=0.0, clamp_max=0.0`
(clamp disabled) and `iter_times=20`. Setting `clamp_min != 0.0` or
`clamp_max != 0.0` enables the clamp stage on both forward and backward.

## Shape assumptions

- `hcMult = 4` — the fast path assumes 4 heads. `mhc_post` also has a
  `_generic` Triton kernel that supports arbitrary small `N`; the other
  three operators currently only implement `hcMult=4`.
- `mhc_post` / `mhc_post_backward` accept both `(B,S,N,D)` (BSND) and
  `(T,N,D)` (TND) inputs; the kernel flattens internally with `T = B*S`
  and reshapes the output back.
- `mhc_pre_clamp_sinkhorn` accepts `(B,S,N,D)` or `(T,N,D)` for `x`. Its
  forward output `y` is `(B,S,D)` or `(T,D)` respectively (the `hin`
  reduction drops the `hcMult` axis). `post_out` and `comb_frag` are
  always `(T, hcMult)` and `(T, hcMult, hcMult)`.
- `phi` is `(hcMix, hcMult*D)` with `hcMix = hcMult*(hcMult+2)`.
- `alpha` is `(3,)` (one scale per head); `base` is `(hcMix,)`.

## Running the correctness harness

```bash
cd FlagOS-mhc
python -m tests.test_correctness
```

Or with `src/` on the path:

```bash
cd FlagOS-mhc
PYTHONPATH=src python tests/test_correctness.py
```

The harness auto-detects device (NPU / CUDA / CPU) and runs four tests:

| Test                                    | What it checks                                        |
|-----------------------------------------|-------------------------------------------------------|
| `mhc_post`                              | `y` vs `mhc_post_ref`                                 |
| `mhc_post_backward`                     | `grad_x, grad_h_res, grad_h_out, grad_h_post` vs ref  |
| `mhc_pre_clamp_sinkhorn`                | `y, post_out, comb_frag` vs ref                       |
| `mhc_pre_clamp_sinkhorn_backward`       | `grad_x, grad_phi, grad_alpha, grad_base` vs ref      |

Numerical tolerances are set loosely for bf16 inputs (1e-2 for the
forward, 5e-2 for the small-stage backward path).

## What is *not* covered vs the production AscendC kernels

- **Deterministic mode** and **Kahan compensated summation** in
  `mhc_pre_clamp_sinkhorn_backward` — the AscendC operator has a
  deterministic path with Kahan accumulation. Not replicated here;
  `torch.mm` and per-program reductions are used instead. If bit-exact
  determinism is required, replace the two `torch.mm` calls with a
  deterministic reduction kernel.
- **AIC (Cube) offload for RMSNorm + matmul stage**. The production
  kernel uses `MIX_AIC_1_2` to overlap AIC (Cube) MMAD with AIV vector
  work; this Triton version merges everything on AIV plus one library
  matmul, which is simpler but may leave Cube underutilised on very
  large `hcMix * HC_D`.
- **hcMult != 4** for `mhc_pre_clamp_sinkhorn` and its backward
  (`mhc_post` already has a generic N kernel).

Both gaps are documented at the top of each source file.
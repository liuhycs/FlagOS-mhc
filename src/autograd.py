"""torch.autograd.Function wrappers to expose forward+backward pairs
in the usual PyTorch style."""

from __future__ import annotations

import torch

from .mhc_post import mhc_post
from .mhc_post_backward import mhc_post_backward
from .mhc_pre_clamp_sinkhorn import mhc_pre_clamp_sinkhorn
from .mhc_pre_clamp_sinkhorn_backward import mhc_pre_clamp_sinkhorn_backward


class MhcPostFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h_res, h_out, h_post):
        y = mhc_post(x, h_res, h_out, h_post)
        ctx.save_for_backward(x, h_res, h_out, h_post)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, h_res, h_out, h_post = ctx.saved_tensors
        gx, ghres, ghout, ghpost = mhc_post_backward(
            grad_y.contiguous(), x, h_res, h_out, h_post
        )
        return gx, ghres, ghout, ghpost


def mhc_post_fn(x, h_res, h_out, h_post):
    """Autograd-aware mhc_post."""
    return MhcPostFn.apply(x, h_res, h_out, h_post)


class MhcPreClampSinkhornFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, phi, alpha, base,
        norm_eps, hc_eps, clamp_min, clamp_max, iter_times,
    ):
        out = mhc_pre_clamp_sinkhorn(
            x, phi, alpha, base,
            norm_eps=norm_eps, hc_eps=hc_eps,
            clamp_min=clamp_min, clamp_max=clamp_max,
            iter_times=iter_times,
            need_backward=True,
        )
        ctx.save_for_backward(
            x, phi, alpha, base,
            out["inv_rms"], out["x_scaled"], out["mixes"],
            out["h_res_logits"], out["pre"],
        )
        ctx.norm_eps = norm_eps
        ctx.hc_eps = hc_eps
        ctx.clamp_min = clamp_min
        ctx.clamp_max = clamp_max
        ctx.iter_times = iter_times
        return out["y"], out["post_out"], out["comb_frag"]

    @staticmethod
    def backward(ctx, grad_y, grad_post_out, grad_comb_frag):
        (x, phi, alpha, base,
         inv_rms, x_scaled, mixes, h_res_logits, pre) = ctx.saved_tensors
        gx, gphi, galpha, gbase = mhc_pre_clamp_sinkhorn_backward(
            x, phi, alpha, base,
            inv_rms, x_scaled, mixes, h_res_logits, pre,
            grad_y.contiguous(), grad_post_out.contiguous(),
            grad_comb_frag.contiguous(),
            norm_eps=ctx.norm_eps, hc_eps=ctx.hc_eps,
            clamp_min=ctx.clamp_min, clamp_max=ctx.clamp_max,
            iter_times=ctx.iter_times,
        )
        # (x, phi, alpha, base, norm_eps, hc_eps, clamp_min, clamp_max, iter_times)
        return gx, gphi, galpha, gbase, None, None, None, None, None


def mhc_pre_clamp_sinkhorn_fn(
    x, phi, alpha, base,
    norm_eps=1e-6, hc_eps=1e-6,
    clamp_min=0.0, clamp_max=0.0,
    iter_times=20,
):
    """Autograd-aware mhc_pre_clamp_sinkhorn. Returns (y, post_out, comb_frag)."""
    return MhcPreClampSinkhornFn.apply(
        x, phi, alpha, base,
        norm_eps, hc_eps, clamp_min, clamp_max, iter_times,
    )

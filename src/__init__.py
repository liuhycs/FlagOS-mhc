"""FlagGems-style Triton implementations of MHC operators, tuned for Ascend NPU.

Four operators are provided (each with a Triton fast path + a PyTorch reference
suffixed `_ref`):

    mhc_post                       (forward)
    mhc_post_backward
    mhc_pre_clamp_sinkhorn         (forward, returns dict)
    mhc_pre_clamp_sinkhorn_backward

Autograd-aware wrappers (torch.autograd.Function):
    mhc_post_fn
    mhc_pre_clamp_sinkhorn_fn
"""

from .mhc_post import mhc_post, mhc_post_ref
from .mhc_post_backward import mhc_post_backward, mhc_post_backward_ref
from .mhc_pre_clamp_sinkhorn import (
    mhc_pre_clamp_sinkhorn,
    mhc_pre_clamp_sinkhorn_ref,
)
from .mhc_pre_clamp_sinkhorn_backward import (
    mhc_pre_clamp_sinkhorn_backward,
    mhc_pre_clamp_sinkhorn_backward_ref,
)
from .autograd import (
    MhcPostFn,
    MhcPreClampSinkhornFn,
    mhc_post_fn,
    mhc_pre_clamp_sinkhorn_fn,
)


__all__ = [
    "mhc_post",
    "mhc_post_ref",
    "mhc_post_backward",
    "mhc_post_backward_ref",
    "mhc_pre_clamp_sinkhorn",
    "mhc_pre_clamp_sinkhorn_ref",
    "mhc_pre_clamp_sinkhorn_backward",
    "mhc_pre_clamp_sinkhorn_backward_ref",
    "MhcPostFn",
    "MhcPreClampSinkhornFn",
    "mhc_post_fn",
    "mhc_pre_clamp_sinkhorn_fn",
]

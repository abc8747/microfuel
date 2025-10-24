from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Generator, Sequence

    from triton.runtime import Autotuner

logger = getLogger(__name__)


def get_autotuner(
    kernels: Sequence,
) -> Generator[Autotuner, None, None]:
    from triton.runtime import Autotuner, Heuristics

    for kernel in kernels:
        if isinstance(kernel, Heuristics):
            kernel = kernel.fn
        assert isinstance(kernel, Autotuner), (
            f"expected {kernel} to be `Autotuner`, got {type(kernel)}"
        )
        yield kernel


def kernels_with_nb():
    from fla.modules import convolution, fused_norm_gate, l2norm

    return (
        # `NB = triton.cdiv(B*T, 1024)`, `B=1` for varlen.
        # NOTE: `bt: tl.constexpr` (block size for sequence dim) still causes recompilation
        convolution.causal_conv1d_fwd_kernel,
        convolution.causal_conv1d_bwd_kernel,
        # `NB = triton.cdiv(T, 2048)`, where `T` is `total_tokens - num_heads`
        # NOTE: `t: tl.constexpr` still causes recompilation.
        l2norm.l2norm_fwd_kernel,
        l2norm.l2norm_bwd_kernel,
        fused_norm_gate.layer_norm_gated_fwd_kernel,
        fused_norm_gate.layer_norm_gated_bwd_kernel,
    )


def fla_autotuner_remove_nb() -> None:
    """[prc25.dataloader.collate_fn][] concatenates variable-length sequences into a single tensor
    `x`. the total number of tokens in `x` (`T` in kernel code) changes with every batch.

    Several `fla` kernels use this total token count to calculate `NB` (number of blocks),
    which is part of their `@triton.autotune` key. a new key triggers recompilation.

    This function removes the `NB` key from the autotuner keys of affected kernels,
    cutting cold start runtime by half.
    """
    for kernel in get_autotuner(kernels_with_nb()):
        kernel.keys = [k for k in kernel.keys if k != "NB"]


def fla_autotuner_check_removed_nb() -> None:
    for kernel in get_autotuner(kernels_with_nb()):
        if "NB" not in kernel.keys:
            continue
        logger.error(
            f"`NB` (number of blocks) still in autotuner keys for {kernel}: {kernel.keys}\n"
            "triton will repeatedly re-autotune this kernel for different sequence lengths, "
            f"please patch the kernel with {fla_autotuner_remove_nb.__name__}!"
        )

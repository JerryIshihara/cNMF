"""Fused CUDA sweep for scikit-learn-compatible Fast-HALS.

This module is imported lazily by :mod:`cnmf.nmf_gpu`; importing cNMF itself
does not require Triton.  One kernel program owns one replicate and one block
of rows.  Replicates and rows are parallel, while the component and gradient
loops remain sequential in the same order as scikit-learn's CD solver.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _hals_sweep_kernel(
    factor,
    gram,
    cross,
    permutation,
    active,
    block_violation,
    M: tl.constexpr,
    K: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    IS_FP64: tl.constexpr,
):
    program = tl.program_id(0)
    replicate = program // N_BLOCKS
    row_block = program - replicate * N_BLOCKS
    rows = row_block * BLOCK + tl.arange(0, BLOCK)
    replicate_offset = replicate.to(tl.int64)
    rows_offset = rows.to(tl.int64)
    row_mask = rows < M
    is_active = tl.load(active + replicate)

    accumulator_dtype = tl.float64 if IS_FP64 else tl.float32
    violation = tl.zeros((BLOCK,), dtype=accumulator_dtype)

    # Gauss-Seidel order is intentional: a component update must be visible to
    # every later component in this same sweep.
    for coordinate in range(0, K):
        component = tl.load(
            permutation + replicate_offset * K + coordinate
        )
        cross_offset = (
            (replicate_offset * K + component) * M + rows_offset
        )
        gradient = -tl.load(cross + cross_offset, mask=row_mask, other=0.0)

        for other_component in range(0, K):
            gram_value = tl.load(
                gram
                + (replicate_offset * K + component) * K
                + other_component
            )
            factor_value = tl.load(
                factor
                + (replicate_offset * K + other_component) * M
                + rows_offset,
                mask=row_mask,
                other=0.0,
            )
            gradient += gram_value * factor_value

        factor_offset = (
            (replicate_offset * K + component) * M + rows_offset
        )
        old_value = tl.load(factor + factor_offset, mask=row_mask, other=0.0)
        projected_gradient = tl.where(
            old_value == 0.0, tl.minimum(0.0, gradient), gradient
        )
        violation += tl.where(
            row_mask & (is_active != 0), tl.abs(projected_gradient), 0.0
        )

        hessian = tl.load(
            gram + (replicate_offset * K + component) * K + component
        )
        # tl.where evaluates both branches, so use 1 only for the division and
        # then select the old value when sklearn's exact hessian guard is false.
        safe_hessian = tl.where(hessian != 0.0, hessian, 1.0)
        candidate = tl.maximum(old_value - gradient / safe_hessian, 0.0)
        new_value = tl.where(hessian != 0.0, candidate, old_value)
        tl.store(
            factor + factor_offset,
            new_value,
            mask=row_mask & (is_active != 0),
        )

    tl.store(
        block_violation
        + replicate_offset * N_BLOCKS
        + row_block.to(tl.int64),
        tl.sum(violation, axis=0),
    )


def hals_sweep_cuda(factor, gram, cross, permutation, active):
    """Update one factor in-place and return projected-gradient violation.

    Parameters use the transposed factor layout ``[replicate, component, row]``.
    ``gram`` is ``[replicate, component, component]`` and ``cross`` has the
    same shape as ``factor``.
    """
    if not factor.is_cuda:
        raise ValueError("hals_sweep_cuda requires CUDA tensors")
    if not factor.is_floating_point() or factor.element_size() not in (4, 8):
        raise TypeError("CUDA Fast-HALS supports fp32 and fp64 tensors")
    if factor.ndim != 3:
        raise ValueError("factor must have shape [replicate, component, row]")

    replicates, components, rows = factor.shape
    expected_factor_shape = (replicates, components, rows)
    if cross.shape != expected_factor_shape:
        raise ValueError("factor and cross must share [replicate, component, row] shape")
    if gram.shape != (replicates, components, components):
        raise ValueError("gram must have shape [replicate, component, component]")
    if permutation.shape != (replicates, components):
        raise ValueError("permutation must have shape [replicate, component]")
    if active.shape != (replicates,):
        raise ValueError("active must have one entry per replicate")
    if permutation.dtype not in (torch.int32, torch.int64):
        raise TypeError("permutation must use an integer dtype")
    if active.dtype != torch.bool:
        raise TypeError("active must use boolean dtype")
    tensors = (factor, gram, cross, permutation, active)
    if any(tensor.device != factor.device for tensor in tensors):
        raise ValueError("all Fast-HALS tensors must be on the same CUDA device")
    if gram.dtype != factor.dtype or cross.dtype != factor.dtype:
        raise TypeError("factor, gram, and cross must share dtype")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Fast-HALS CUDA tensors must be contiguous")

    block = 128
    n_blocks = triton.cdiv(rows, block)
    block_violation = factor.new_empty((replicates, n_blocks))
    grid = (replicates * n_blocks,)
    _hals_sweep_kernel[grid](
        factor,
        gram,
        cross,
        permutation,
        active,
        block_violation,
        M=rows,
        K=components,
        N_BLOCKS=n_blocks,
        BLOCK=block,
        IS_FP64=factor.element_size() == 8,
        num_warps=4,
    )
    return block_violation.sum(dim=1)

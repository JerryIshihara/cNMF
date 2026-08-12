"""Batched sklearn-compatible Fast-HALS coordinate-descent NMF solver."""

import numpy as np

from . import utils


# ---------------------------------------------------------------------
# CD solver (sklearn-compatible Fast-HALS; batch-aware)
# ---------------------------------------------------------------------


def _numpy_staging_dtype(torch, dtype):
    """Return the host dtype used for sklearn initialization before transfer."""
    if dtype is torch.float64:
        return np.float64
    if dtype is torch.float32:
        return np.float32
    raise TypeError("solver='cd' supports gpu dtype fp32 or fp64")


def _to_checked_custom_factor(value, shape, name, dtype):
    """Validate and cast a custom CD factor."""
    if value is None:
        raise ValueError(f"init='custom' requires {name}")
    array = value.toarray() if hasattr(value, "toarray") else np.asarray(value)
    if array.ndim != 2 or array.shape != shape:
        raise ValueError(f"custom {name} shape must be {shape}; got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"custom {name} contains NaN/inf")
    if array.size and array.min() < 0:
        raise ValueError(f"custom {name} must be non-negative")
    return np.ascontiguousarray(array, dtype=dtype)


def _cd_regularization(nmf_kwargs, n_samples, n_features):
    """Return sklearn's sample/feature-scaled CD regularization terms."""
    alpha_w = float(nmf_kwargs.get("alpha_W", 0.0))
    alpha_h_raw = nmf_kwargs.get("alpha_H", "same")
    alpha_h = alpha_w if alpha_h_raw == "same" else float(alpha_h_raw)
    l1_ratio = float(nmf_kwargs.get("l1_ratio", 0.0))
    if alpha_w < 0 or alpha_h < 0:
        raise ValueError("alpha_W and alpha_H must be non-negative")
    if not 0.0 <= l1_ratio <= 1.0:
        raise ValueError("l1_ratio must be in the range [0, 1]")
    return (
        n_features * alpha_w * l1_ratio,
        n_features * alpha_w * (1.0 - l1_ratio),
        n_samples * alpha_h * l1_ratio,
        n_samples * alpha_h * (1.0 - l1_ratio),
    )


def _validate_cd_runtime(torch, rc, nmf_kwargs):
    """Validate the sklearn CD contract before allocating factor tensors."""
    legacy_regularization = {"alpha", "regularization"}.intersection(nmf_kwargs)
    if legacy_regularization:
        raise ValueError(
            "solver='cd' does not accept deprecated alpha/regularization; "
            "use alpha_W, alpha_H, and l1_ratio"
        )
    beta_loss = nmf_kwargs.get("beta_loss", "frobenius")
    if not (beta_loss == 2 or str(beta_loss).lower() == "frobenius"):
        raise ValueError("solver='cd' supports only beta_loss='frobenius'")
    if rc.dtype is torch.bfloat16:
        raise ValueError("solver='cd' supports gpu dtype fp32 or fp64, not bf16")
    if rc.max_iter < 1:
        raise ValueError("max_iter must be at least 1 for solver='cd'")
    if rc.tol < 0:
        raise ValueError("tol must be non-negative for solver='cd'")
    if not isinstance(nmf_kwargs.get("shuffle", False), (bool, np.bool_)):
        raise ValueError("shuffle must be a boolean for solver='cd'")


def _hals_sweep_torch(
    factor, gram, cross, permutation, active, return_row_violation=False
):
    """Apply one literal torch port of sklearn's serial CD coordinate sweep."""
    replicates, components, rows = factor.shape
    violation = factor.new_zeros(
        (replicates, rows) if return_row_violation else replicates
    )
    zero = factor.new_zeros(())
    cyclic = permutation is None

    for coordinate in range(components):
        if cyclic:
            component = coordinate
            gram_row = gram[:, component, :]
            cross_row = cross[:, component, :]
            old_value = factor[:, component, :].clone()
        else:
            component = permutation[:, coordinate]
            gram_row = gram.gather(
                1, component[:, None, None].expand(-1, 1, components)
            ).squeeze(1)
            factor_index = component[:, None, None].expand(-1, 1, rows)
            cross_row = cross.gather(1, factor_index).squeeze(1)
            old_value = factor.gather(1, factor_index).squeeze(1)

        # Match sklearn's _cdnmf_fast.pyx summation and Gauss-Seidel order.
        gradient = -cross_row.clone()
        for other_component in range(components):
            gradient = (
                gradient
                + gram_row[:, other_component, None]
                * factor[:, other_component, :]
            )

        projected_gradient = gradient.where(
            old_value != 0, zero.minimum(gradient)
        )
        if return_row_violation:
            violation = (
                violation
                + projected_gradient.abs() * active[:, None]
            )
        else:
            violation = (
                violation
                + projected_gradient.abs().sum(dim=1) * active
            )

        if cyclic:
            hessian = gram[:, component, component, None]
        else:
            hessian = gram_row.gather(1, component[:, None])
        nonzero_hessian = hessian != 0
        safe_hessian = hessian.where(nonzero_hessian, hessian.new_ones(()))
        candidate = (old_value - gradient / safe_hessian).clamp_min(0)
        new_value = candidate.where(nonzero_hessian, old_value)
        new_value = new_value.where(active[:, None], old_value)

        if cyclic:
            factor[:, component, :] = new_value
        else:
            factor.scatter_(1, factor_index, new_value[:, None, :])

    return violation


_HALS_CUDA_BACKEND_UNSET = object()
_HALS_CUDA_BACKEND = _HALS_CUDA_BACKEND_UNSET


def _get_hals_cuda_backend():
    """Resolve and cache the optional fused CUDA sweep."""
    global _HALS_CUDA_BACKEND
    if _HALS_CUDA_BACKEND is _HALS_CUDA_BACKEND_UNSET:
        try:
            from .solver_cd_triton import hals_sweep_cuda
        except (ImportError, ModuleNotFoundError):
            hals_sweep_cuda = None
        _HALS_CUDA_BACKEND = hals_sweep_cuda
    return _HALS_CUDA_BACKEND


def _hals_sweep(
    factor, gram, cross, permutation, active, return_row_violation=False
):
    """Use the fused CUDA sweep when available, otherwise use torch."""
    if factor.is_cuda:
        hals_sweep_cuda = _get_hals_cuda_backend()
        if hals_sweep_cuda is not None:
            if permutation is None:
                permutation = factor.new_tensor(
                    np.tile(
                        np.arange(factor.shape[1], dtype=np.int64),
                        (factor.shape[0], 1),
                    )
                ).long()
            return hals_sweep_cuda(
                factor,
                gram,
                cross,
                permutation,
                active,
                return_row_violation=return_row_violation,
            )
    return _hals_sweep_torch(
        factor,
        gram,
        cross,
        permutation,
        active,
        return_row_violation=return_row_violation,
    )


def _regularize_cd_products(gram, cross, l1_reg, l2_reg):
    """Apply sklearn's L2 diagonal addition and L1 cross-product shift."""
    if l2_reg != 0.0:
        gram.diagonal(dim1=-2, dim2=-1).add_(l2_reg)
    if l1_reg != 0.0:
        cross.sub_(l1_reg)
    return gram, cross


def _batch_invariant_cd_products(factor, data):
    """Build CD products with one identical 2D GEMM path per replicate.

    A single batched matmul may select a different fp32 CUDA reduction path as
    the replicate count changes.  Fast-HALS amplifies those small differences,
    which can change projected-gradient stopping decisions.  Keep the factor
    state and HALS sweep batched, but compute each replicate's products through
    the same two-dimensional matmul shape used by a batch of one.
    """
    replicates, components, _ = factor.shape
    gram = factor.new_empty((replicates, components, components))
    cross = factor.new_empty((replicates, components, data.shape[-1]))
    for replicate in range(replicates):
        replicate_factor = factor[replicate]
        gram[replicate].copy_(
            replicate_factor @ replicate_factor.transpose(-2, -1)
        )
        cross[replicate].copy_(replicate_factor @ data)
    return gram, cross


def _fit_cd(torch, rc, Xg, Wt, H, update_h, seeds, nmf_kwargs):
    """Run batched sklearn-compatible Fast-HALS to projected-gradient convergence."""
    max_iter = rc.max_iter
    tol = rc.tol
    regularization = _cd_regularization(
        nmf_kwargs, Xg.shape[0], Xg.shape[1]
    )
    shuffle = bool(nmf_kwargs.get("shuffle", False))
    tf32 = utils._want_tf32(torch, rc)
    device = rc.device
    if Xg.dtype != Wt.dtype or Xg.dtype != H.dtype:
        raise RuntimeError(
            "NMF runtime tensors must share dtype; got "
            f"{sorted(map(str, {Xg.dtype, Wt.dtype, H.dtype}))}"
        )

    replicates, components, _ = Wt.shape
    active = torch.ones(replicates, dtype=torch.bool, device=device)
    n_iter = torch.zeros(replicates, dtype=torch.int64, device=device)
    violation_init = Wt.new_zeros(replicates)
    l1_w, l2_w, l1_h, l2_h = regularization

    if shuffle:
        try:
            from sklearn.utils import check_random_state
        except ModuleNotFoundError as exc:
            raise RuntimeError("scikit-learn is required for shuffled CD") from exc
        rngs = [check_random_state(seed) for seed in seeds]
        cyclic_permutation = None
    else:
        rngs = None
        cyclic_permutation = (
            torch.arange(components, dtype=torch.int64, device=device)
            .expand(replicates, -1)
            .contiguous()
            if Xg.is_cuda and _get_hals_cuda_backend() is not None
            else None
        )

    def next_permutation():
        if rngs is None:
            return cyclic_permutation
        values = np.stack(
            [rng.permutation(components) for rng in rngs], axis=0
        ).astype(np.int64, copy=False)
        return torch.as_tensor(values, dtype=torch.int64, device=device)

    with torch.no_grad(), utils._cuda_tf32(torch, tf32, device):
        for iteration in range(1, max_iter + 1):
            gram, cross = _batch_invariant_cd_products(
                H, Xg.transpose(-2, -1)
            )
            gram, cross = _regularize_cd_products(gram, cross, l1_w, l2_w)
            violation = _hals_sweep(
                Wt, gram, cross, next_permutation(), active
            )

            if update_h:
                gram, cross = _batch_invariant_cd_products(Wt, Xg)
                gram, cross = _regularize_cd_products(gram, cross, l1_h, l2_h)
                violation = violation + _hals_sweep(
                    H, gram, cross, next_permutation(), active
                )

            n_iter = n_iter.new_full((), iteration).where(active, n_iter)
            if iteration == 1:
                violation_init = violation.clone()

            zero_init = violation_init == 0
            denominator = violation_init.where(
                ~zero_init, violation_init.new_ones(())
            )
            converged = active & (
                zero_init | ((violation / denominator) <= tol)
            )
            active = active & ~converged
            if not bool(active.any()):
                break

    return Wt, H, n_iter



def _fit_cd_fixed_h_tiled(
    torch, rc, Xcompute, H, row_batch, seeds, nmf_kwargs
):
    """Update W in row tiles from one shared set of fixed-H products."""
    max_iter = rc.max_iter
    tol = rc.tol
    regularization = _cd_regularization(
        nmf_kwargs, Xcompute.shape[0], Xcompute.shape[1]
    )
    shuffle = bool(nmf_kwargs.get("shuffle", False))
    tf32 = utils._want_tf32(torch, rc)
    device = rc.device
    replicates, components, _ = H.shape
    n_rows = Xcompute.shape[0]
    Wt_host = np.zeros(
        (replicates, components, n_rows), dtype=Xcompute.dtype
    )
    active = torch.ones(replicates, dtype=torch.bool, device=device)
    n_iter = torch.zeros(replicates, dtype=torch.int64, device=device)
    violation_init = H.new_zeros(replicates)
    l1_w, l2_w, _l1_h, _l2_h = regularization

    # H is fixed, so H @ H.T and H @ X.T are invariant across iterations.
    # Build them once on the host for both tiled and untiled execution. This
    # makes the arithmetic inputs independent of the requested row partition
    # and avoids transferring/reducing the multi-gigabyte X every iteration.
    Hcompute = H.cpu().numpy()
    gram_host = np.empty(
        (replicates, components, components), dtype=Xcompute.dtype
    )
    cross_host = np.empty(
        (replicates, components, n_rows), dtype=Xcompute.dtype
    )
    for replicate in range(replicates):
        gram_host[replicate] = Hcompute[replicate] @ Hcompute[replicate].T
        cross_host[replicate] = Hcompute[replicate] @ Xcompute.T
    if l1_w != 0.0:
        cross_host -= l1_w
    gram = torch.as_tensor(gram_host, dtype=H.dtype, device=device)
    if l2_w != 0.0:
        gram.diagonal(dim1=-2, dim2=-1).add_(l2_w)

    if shuffle:
        try:
            from sklearn.utils import check_random_state
        except ModuleNotFoundError as exc:
            raise RuntimeError("scikit-learn is required for shuffled CD") from exc
        rngs = [check_random_state(seed) for seed in seeds]
        cyclic_permutation = None
    else:
        rngs = None
        cyclic_permutation = (
            torch.arange(components, dtype=torch.int64, device=device)
            .expand(replicates, -1)
            .contiguous()
            if H.is_cuda and _get_hals_cuda_backend() is not None
            else None
        )

    def next_permutation():
        if rngs is None:
            return cyclic_permutation
        values = np.stack(
            [rng.permutation(components) for rng in rngs], axis=0
        ).astype(np.int64, copy=False)
        return torch.as_tensor(values, dtype=torch.int64, device=device)

    with torch.no_grad(), utils._cuda_tf32(torch, tf32, device):
        for iteration in range(1, max_iter + 1):
            permutation = next_permutation()
            row_violation_host = np.empty(
                (replicates, n_rows), dtype=Xcompute.dtype
            )

            for start in range(0, n_rows, row_batch):
                stop = min(start + row_batch, n_rows)
                Wt = torch.as_tensor(
                    np.ascontiguousarray(Wt_host[:, :, start:stop]),
                    dtype=H.dtype,
                    device=device,
                )
                cross = torch.as_tensor(
                    np.ascontiguousarray(cross_host[:, :, start:stop]),
                    dtype=H.dtype,
                    device=device,
                )
                row_violation = _hals_sweep(
                    Wt,
                    gram,
                    cross,
                    permutation,
                    active,
                    return_row_violation=True,
                )
                row_violation_host[:, start:stop] = (
                    row_violation.cpu().numpy()
                )
                Wt_host[:, :, start:stop] = Wt.cpu().numpy()

            # Reduce once in global row order so the convergence decision is
            # independent of where the requested row tiles begin and end.
            violation = torch.as_tensor(
                row_violation_host, dtype=H.dtype, device=device
            ).sum(dim=1)

            n_iter = n_iter.new_full((), iteration).where(active, n_iter)
            if iteration == 1:
                violation_init = violation.clone()

            zero_init = violation_init == 0
            denominator = violation_init.where(
                ~zero_init, violation_init.new_ones(())
            )
            converged = active & (
                zero_init | ((violation / denominator) <= tol)
            )
            active = active & ~converged
            if not bool(active.any()):
                break

    return Wt_host, n_iter


# ---------------------------------------------------------------------
# CD solver entry points
# ---------------------------------------------------------------------


def _nmf_gpu_cd_update_h(torch, rc, seeds, nmf_kwargs):
    """Run CD with both W and H updated."""
    host_dtype = _numpy_staging_dtype(torch, rc.dtype)
    Xcompute = np.ascontiguousarray(rc.Xnp, dtype=host_dtype)
    Xg = torch.as_tensor(Xcompute, dtype=rc.dtype, device=rc.device)
    replicates = len(seeds)

    init = nmf_kwargs.get("init")
    if init == "custom":
        W0 = _to_checked_custom_factor(
            nmf_kwargs.get("W"),
            (Xcompute.shape[0], rc.k),
            "W",
            host_dtype,
        )
        H0 = _to_checked_custom_factor(
            nmf_kwargs.get("H"),
            (rc.k, Xcompute.shape[1]),
            "H",
            host_dtype,
        )
        Wt0 = np.repeat(W0.T[None, :, :], replicates, axis=0)
        Hs0 = np.repeat(H0[None, :, :], replicates, axis=0)
    else:
        Wt0 = np.empty(
            (replicates, rc.k, Xcompute.shape[0]), dtype=host_dtype
        )
        Hs0 = np.empty(
            (replicates, rc.k, Xcompute.shape[1]), dtype=host_dtype
        )
        for replicate, seed in enumerate(seeds):
            W0, H0 = utils._init_wh(Xcompute, rc.k, seed, init)
            Wt0[replicate] = W0.T
            Hs0[replicate] = H0

    Wt = torch.as_tensor(Wt0, dtype=rc.dtype, device=rc.device)
    H = torch.as_tensor(Hs0, dtype=rc.dtype, device=rc.device)
    Wt, H, _ = _fit_cd(torch, rc, Xg, Wt, H, True, seeds, nmf_kwargs)

    Hc = H.cpu().double().numpy()
    Wc = Wt.transpose(-2, -1).cpu().double().numpy()
    return [(Hc[r], Wc[r]) for r in range(replicates)]


def _nmf_gpu_cd_fixed_h(torch, rc, seeds, nmf_kwargs):
    """Run CD with H fixed and only W updated."""
    host_dtype = _numpy_staging_dtype(torch, rc.dtype)
    Xcompute = np.ascontiguousarray(rc.Xnp, dtype=host_dtype)
    replicates = len(seeds)

    H0 = utils._to_checked_fixed_h(
        nmf_kwargs.get("H"), rc.k, Xcompute.shape[1]
    )
    H0 = np.ascontiguousarray(H0, dtype=host_dtype)
    H = (
        torch.as_tensor(H0, dtype=rc.dtype, device=rc.device)
        .unsqueeze(0)
        .expand(replicates, -1, -1)
        .contiguous()
    )

    row_batch, tiling_ratio = utils.resolve_row_batch(
        torch,
        rc.device,
        Xcompute,
        replicates,
        rc.k,
        configured_ratio=rc.opt["row_tiling_ratio"],
    )

    Wt_host, _ = _fit_cd_fixed_h_tiled(
        torch,
        rc,
        Xcompute,
        H,
        row_batch,
        seeds,
        nmf_kwargs,
    )
    Wc = Wt_host.transpose(0, 2, 1).astype(np.float64, copy=False)

    Hc = H.cpu().double().numpy()
    return [(Hc[r], Wc[r]) for r in range(replicates)]


def _nmf_gpu_cd(X, seeds, nmf_kwargs, gpu_kwargs=None):
    """Dispatch full or fixed-H sklearn-compatible CD replicates."""
    torch = utils._loud_import_torch()
    seeds = utils._normalize_seeds(seeds)
    rc = utils._gpu_setup(torch, X, nmf_kwargs, gpu_kwargs)
    _validate_cd_runtime(torch, rc, nmf_kwargs)

    if nmf_kwargs.get("update_H", True) is not False:
        return _nmf_gpu_cd_update_h(torch, rc, seeds, nmf_kwargs)
    return _nmf_gpu_cd_fixed_h(torch, rc, seeds, nmf_kwargs)

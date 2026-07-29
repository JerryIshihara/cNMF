"""Controlled NMF backend benchmarking helpers.

This module deliberately reuses the implementations that cNMF uses:

* CPU CD calls :meth:`cNMF._nmf`, which delegates to sklearn.
* CPU MU advances sklearn's own Frobenius multiplicative-update functions so
  checkpoints can be captured without restarting the optimization.
* GPU CD advances cNMF's production Fast-HALS sweeps from persisted batched
  factors and records one raw CUDA-event duration per outer iteration.
* GPU MU calls cNMF's ``_mu_step`` kernel, including the sklearn-compatible
  exact-zero denominator replacement.
* Torch NMF uses ``torchnmf`` with its denominator handling adapted to the
  same exact-zero rule. The dependency remains optional and is imported only
  when that backend is requested.

All entry points accept explicit W/H factors. Random seeds are metadata once
those factors exist; no backend is allowed to regenerate an initialization.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from .cnmf import cNMF
from .nmf_gpu import (
    DEFAULT_GPU,
    _cuda_tf32,
    _cd_regularization,
    _get_hals_cuda_backend,
    _hals_sweep,
    _loud_import_torch,
    _mu_step,
    _regularize_cd_products,
    _select_device,
    _select_storage,
    _to_checked_array,
)


CheckpointCallback = Callable[[int, np.ndarray, np.ndarray], None]
GPUCDCheckpointCallback = Callable[[int, Any, Any, Any, dict[str, Any]], None]
SKLEARN_EXACT_ZERO_EPSILON = float(np.finfo(np.float32).eps)


def _validated_checkpoints(checkpoints: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in checkpoints)
    if not values or any(value < 1 for value in values):
        raise ValueError("checkpoints must contain positive iteration numbers")
    if tuple(sorted(set(values))) != values:
        raise ValueError("checkpoints must be unique and strictly increasing")
    return values


def _validated_factors(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    *,
    batched: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = _to_checked_array(X)
    W = np.asarray(W)
    H = np.asarray(H)

    expected_w_tail = (X.shape[0], H.shape[-2])
    expected_h_tail = (H.shape[-2], X.shape[1])
    if batched:
        if W.ndim != 3 or H.ndim != 3 or W.shape[0] != H.shape[0]:
            raise ValueError("batched W/H must have shapes [R,n,k] and [R,k,g]")
        if W.shape[1:] != expected_w_tail or H.shape[1:] != expected_h_tail:
            raise ValueError(
                f"batched factor shapes do not match X: X={X.shape}, "
                f"W={W.shape}, H={H.shape}"
            )
    else:
        if W.ndim != 2 or H.ndim != 2:
            raise ValueError("W/H must have shapes [n,k] and [k,g]")
        if W.shape != expected_w_tail or H.shape != expected_h_tail:
            raise ValueError(
                f"factor shapes do not match X: X={X.shape}, W={W.shape}, H={H.shape}"
            )

    for name, value in (("W", W), ("H", H)):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN/inf")
        if value.size and value.min() < 0:
            raise ValueError(f"{name} must be non-negative")
    if X.dtype != W.dtype or X.dtype != H.dtype:
        raise ValueError(
            f"X, W, and H must share a dtype; got {X.dtype}, {W.dtype}, {H.dtype}"
        )
    return X, np.ascontiguousarray(W), np.ascontiguousarray(H)


def run_cpu_cd(
    X: np.ndarray,
    W0: np.ndarray,
    H0: np.ndarray,
    *,
    seed: int,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run cNMF's sklearn-backed coordinate-descent implementation."""
    X, W, H = _validated_factors(X, W0, H0, batched=False)
    kwargs: dict[str, Any] = {
        "W": W.copy(),
        "H": H.copy(),
        "n_components": H.shape[0],
        "init": "custom",
        "solver": "cd",
        "beta_loss": "frobenius",
        "tol": 0.0,
        "max_iter": int(max_iter),
        "alpha_W": 0.0,
        "alpha_H": 0.0,
        "l1_ratio": 0.0,
        "random_state": int(seed),
        "shuffle": False,
    }
    spectra, usages = cNMF(output_dir=".", name="benchmark")._nmf(X, kwargs)
    return usages, spectra, int(max_iter)


def run_cpu_mu_checkpoints(
    X: np.ndarray,
    W0: np.ndarray,
    H0: np.ndarray,
    *,
    checkpoints: Sequence[int],
    callback: CheckpointCallback,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance sklearn's cNMF CPU-MU path and emit exact iteration checkpoints."""
    from sklearn.decomposition._nmf import (
        _multiplicative_update_h,
        _multiplicative_update_w,
    )

    checkpoints = _validated_checkpoints(checkpoints)
    X, W, H = _validated_factors(X, W0, H0, batched=False)
    W = W.copy()
    H = H.copy()
    wanted = set(checkpoints)

    for iteration in range(1, checkpoints[-1] + 1):
        W, _, _, _ = _multiplicative_update_w(
            X,
            W,
            H,
            beta_loss=2,
            l1_reg_W=0.0,
            l2_reg_W=0.0,
            gamma=1.0,
            update_H=True,
        )
        H = _multiplicative_update_h(
            X,
            W,
            H,
            beta_loss=2,
            l1_reg_H=0.0,
            l2_reg_H=0.0,
            gamma=1.0,
        )
        if iteration in wanted:
            callback(iteration, W, H)
    return W, H


def run_gpu_mu_checkpoints(
    X: np.ndarray,
    W0: np.ndarray,
    H0: np.ndarray,
    *,
    checkpoints: Sequence[int],
    callback: CheckpointCallback,
    device: str = "cuda",
    dtype: str = "fp32",
    allow_tf32: bool = False,
) -> dict[str, float | str | int]:
    """Run the fixed cNMF Torch MU kernel on a batch of explicit initial factors."""
    checkpoints = _validated_checkpoints(checkpoints)
    X, W, H = _validated_factors(X, W0, H0, batched=True)
    torch = _loud_import_torch()
    resolved_device = _select_device(torch, device)
    resolved_dtype = _select_storage(torch, dtype, resolved_device)
    expected_dtype = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
    }.get(X.dtype)
    if expected_dtype is None or resolved_dtype is not expected_dtype:
        raise ValueError(
            f"requested Torch dtype {resolved_dtype} does not match X dtype {X.dtype}"
        )

    torch.use_deterministic_algorithms(True)
    Xg = torch.as_tensor(X, dtype=resolved_dtype, device=resolved_device).unsqueeze(0)
    Wg = torch.as_tensor(W, dtype=resolved_dtype, device=resolved_device)
    Hg = torch.as_tensor(H, dtype=resolved_dtype, device=resolved_device)
    eps = torch.tensor(
        SKLEARN_EXACT_ZERO_EPSILON,
        dtype=resolved_dtype,
        device=resolved_device,
    )

    if resolved_device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    wanted = set(checkpoints)
    with torch.no_grad(), _cuda_tf32(torch, allow_tf32, resolved_device):
        for iteration in range(1, checkpoints[-1] + 1):
            Wg, Hg = _mu_step(Wg, Hg, Xg, eps)
            if iteration in wanted:
                if resolved_device.startswith("cuda"):
                    torch.cuda.synchronize()
                callback(
                    iteration,
                    Wg.detach().cpu().numpy(),
                    Hg.detach().cpu().numpy(),
                )

    metrics: dict[str, float | str | int] = {
        "device": resolved_device,
        "batch_size": int(W.shape[0]),
        "zero_guard": "replace_exact_zero_with_float32_eps",
        "epsilon": SKLEARN_EXACT_ZERO_EPSILON,
    }
    if resolved_device.startswith("cuda"):
        metrics.update(
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        )
    return metrics


def _cd_permutation_factory(
    torch: Any,
    *,
    replicates: int,
    components: int,
    device: str,
    use_cuda_backend: bool,
    shuffle: bool,
    seeds: Sequence[int | None],
):
    """Return sklearn-compatible per-sweep coordinate permutations."""
    if shuffle:
        try:
            from sklearn.utils import check_random_state
        except ModuleNotFoundError as exc:
            raise RuntimeError("scikit-learn is required for shuffled CD") from exc
        rngs = [check_random_state(seed) for seed in seeds]

        def next_permutation():
            values = np.stack(
                [rng.permutation(components) for rng in rngs], axis=0
            ).astype(np.int64, copy=False)
            return torch.as_tensor(
                values, dtype=torch.int64, device=device
            )

        return next_permutation

    # This is the same one-time cyclic permutation used by production
    # ``_fit_cd`` for the fused Triton kernel. The literal Torch fallback uses
    # ``None`` to avoid a gather/scatter path for an already-cyclic ordering.
    cyclic_permutation = (
        torch.arange(components, dtype=torch.int64, device=device)
        .expand(replicates, -1)
        .contiguous()
        if use_cuda_backend
        else None
    )
    return lambda: cyclic_permutation


def _gpu_cd_iteration(
    Wt: Any,
    H: Any,
    Xg: Any,
    active: Any,
    *,
    update_H: bool,
    regularization: tuple[float, float, float, float],
    next_permutation: Callable[[], Any],
):
    """Advance one outer CD iteration using the production Fast-HALS sweep."""
    l1_W, l2_W, l1_H, l2_H = regularization
    gram = H @ H.transpose(-2, -1)
    cross = H @ Xg.transpose(-2, -1)
    gram, cross = _regularize_cd_products(gram, cross, l1_W, l2_W)
    violation = _hals_sweep(
        Wt, gram, cross, next_permutation(), active
    )

    if update_H:
        gram = Wt @ Wt.transpose(-2, -1)
        cross = Wt @ Xg
        gram, cross = _regularize_cd_products(
            gram, cross, l1_H, l2_H
        )
        violation = violation + _hals_sweep(
            H, gram, cross, next_permutation(), active
        )
    return violation


def run_gpu_cd_checkpoints(
    X: np.ndarray,
    W0: np.ndarray,
    H0: np.ndarray,
    *,
    checkpoints: Sequence[int],
    callback: GPUCDCheckpointCallback,
    device: str = "cuda",
    dtype: str = "fp32",
    seeds: Sequence[int | None] | None = None,
    shuffle: bool = False,
    alpha_W: float = 0.0,
    alpha_H: float | str = 0.0,
    l1_ratio: float = 0.0,
    warmup: bool = True,
) -> dict[str, Any]:
    """Benchmark batched production Fast-HALS from persisted W/H factors.

    ``W0`` and ``H0`` must have shapes ``[replicate, sample, component]`` and
    ``[replicate, component, feature]``.  Their NumPy dtype is authoritative:
    the requested Torch dtype must match, which prevents a persisted fp32
    benchmark initialization from being silently regenerated or promoted.

    One CUDA event pair brackets each *individual* outer CD iteration. Event
    pairs and active-count tensors are materialized only at checkpoints, so
    reported timings are raw iteration latencies rather than an amortized
    checkpoint duration. A one-iteration warm-up (when requested) mutates
    clones; the timed run starts from the untouched device copies of W0/H0.

    At each reached checkpoint, ``callback`` receives
    ``(iteration, X_device, W_device, H_device, state)``. ``W_device`` is the
    canonical ``[R,n,k]`` view and all three factors remain on the selected
    device. This lets a runner compute reconstruction errors before making
    any host factor copies. Tensor references are live and will be updated
    after the callback returns; clone anything that must be retained.

    Fast-HALS is run with TF32 disabled and sklearn's ``tol=0`` stopping rule.
    An all-converged batch is observed at the next checkpoint (rather than
    synchronizing CUDA on every iteration), then the run stops.
    """
    checkpoints = _validated_checkpoints(checkpoints)
    if not callable(callback):
        raise TypeError("callback must be callable")
    X, W, H = _validated_factors(X, W0, H0, batched=True)
    torch = _loud_import_torch()
    torch.use_deterministic_algorithms(True)
    resolved_device = _select_device(torch, device)
    resolved_dtype = _select_storage(torch, dtype, resolved_device)
    expected_dtype = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
    }.get(X.dtype)
    if expected_dtype is None or resolved_dtype is not expected_dtype:
        raise ValueError(
            f"requested Torch dtype {resolved_dtype} does not match X dtype {X.dtype}"
        )
    if resolved_dtype not in (torch.float32, torch.float64):
        raise ValueError("Fast-HALS benchmark supports fp32 and fp64 only")

    replicates, samples, components = W.shape
    if seeds is None:
        seeds = [None] * replicates
    else:
        seeds = tuple(
            None if seed is None else int(seed) for seed in seeds
        )
        if len(seeds) != replicates:
            raise ValueError("seeds must contain one value per replicate")

    regularization = _cd_regularization(
        {
            "alpha_W": alpha_W,
            "alpha_H": alpha_H,
            "l1_ratio": l1_ratio,
        },
        samples,
        X.shape[1],
    )
    Xg = torch.as_tensor(
        np.ascontiguousarray(X), dtype=resolved_dtype, device=resolved_device
    )
    # Production CD stores usages transposed as [R,k,n].
    Wt = torch.as_tensor(
        np.array(W.transpose(0, 2, 1), copy=True, order="C"),
        dtype=resolved_dtype,
        device=resolved_device,
    )
    Hg = torch.as_tensor(
        np.array(H, copy=True, order="C"),
        dtype=resolved_dtype,
        device=resolved_device,
    )
    use_cuda = resolved_device.startswith("cuda")
    use_cuda_backend = (
        use_cuda and _get_hals_cuda_backend() is not None
    )

    if use_cuda:
        torch.cuda.synchronize(device=resolved_device)
        torch.cuda.empty_cache()

    with torch.no_grad(), _cuda_tf32(torch, False, resolved_device):
        if warmup:
            warm_Wt = Wt.clone()
            warm_H = Hg.clone()
            warm_active = torch.ones(
                replicates, dtype=torch.bool, device=resolved_device
            )
            warm_permutation = _cd_permutation_factory(
                torch,
                replicates=replicates,
                components=components,
                device=resolved_device,
                use_cuda_backend=use_cuda_backend,
                shuffle=shuffle,
                seeds=seeds,
            )
            _gpu_cd_iteration(
                warm_Wt,
                warm_H,
                Xg,
                warm_active,
                update_H=True,
                regularization=regularization,
                next_permutation=warm_permutation,
            )
            if use_cuda:
                torch.cuda.synchronize(device=resolved_device)
            del warm_Wt, warm_H, warm_active

        # Recreate the RNGs after warm-up. Wt/Hg themselves were never
        # mutated, so both factors and shuffled coordinate streams start from
        # the exact persisted state.
        next_permutation = _cd_permutation_factory(
            torch,
            replicates=replicates,
            components=components,
            device=resolved_device,
            use_cuda_backend=use_cuda_backend,
            shuffle=shuffle,
            seeds=seeds,
        )
        active = torch.ones(
            replicates, dtype=torch.bool, device=resolved_device
        )
        n_iter = torch.zeros(
            replicates, dtype=torch.int64, device=resolved_device
        )
        violation_init = Wt.new_zeros(replicates)
        iteration_seconds: list[float] = []
        active_replicates_per_iteration: list[int] = []
        checkpoint_active_replicates: dict[int, int] = {}
        pending_events: list[tuple[Any, Any]] = []
        pending_active_counts: list[Any] = []
        wanted = set(checkpoints)

        if use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device=resolved_device)

        completed_iterations = 0
        for iteration in range(1, checkpoints[-1] + 1):
            if use_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                started_at = time.perf_counter()

            violation = _gpu_cd_iteration(
                Wt,
                Hg,
                Xg,
                active,
                update_H=True,
                regularization=regularization,
                next_permutation=next_permutation,
            )
            n_iter = n_iter.new_full((), iteration).where(active, n_iter)
            if iteration == 1:
                violation_init = violation.clone()
            zero_init = violation_init == 0
            denominator = violation_init.where(
                ~zero_init, violation_init.new_ones(())
            )
            converged = active & (
                zero_init | ((violation / denominator) <= 0.0)
            )
            active = active & ~converged

            if use_cuda:
                end_event.record()
                pending_events.append((start_event, end_event))
                pending_active_counts.append(active.sum())
            else:
                iteration_seconds.append(time.perf_counter() - started_at)
                active_replicates_per_iteration.append(
                    int(active.sum().item())
                )
            completed_iterations = iteration

            if iteration not in wanted:
                continue

            if use_cuda:
                torch.cuda.synchronize(device=resolved_device)
                iteration_seconds.extend(
                    start.elapsed_time(end) / 1000.0
                    for start, end in pending_events
                )
                active_replicates_per_iteration.extend(
                    int(value)
                    for value in torch.stack(
                        pending_active_counts
                    ).cpu().tolist()
                )
                pending_events.clear()
                pending_active_counts.clear()

            checkpoint_active_replicates[iteration] = (
                active_replicates_per_iteration[-1]
            )
            state = {
                "active": active.clone(),
                "n_iter": n_iter.clone(),
                "active_replicates": checkpoint_active_replicates[iteration],
                "iteration_seconds": tuple(iteration_seconds),
                "last_iteration_seconds": iteration_seconds[-1],
            }
            # Invoke the metric/checkpoint hook while all factors remain on
            # device and before this helper makes any host-side factor copy.
            callback(
                iteration,
                Xg,
                Wt.transpose(-2, -1),
                Hg,
                state,
            )
            if checkpoint_active_replicates[iteration] == 0:
                break

    metrics: dict[str, Any] = {
        "device": resolved_device,
        "dtype": dtype,
        "batch_size": replicates,
        "components": components,
        "checkpoints_requested": list(checkpoints),
        "checkpoints_reached": list(checkpoint_active_replicates),
        "completed_iterations": completed_iterations,
        "iteration_seconds": iteration_seconds,
        "active_replicates_per_iteration": active_replicates_per_iteration,
        "checkpoint_active_replicates": checkpoint_active_replicates,
        "n_iter": n_iter.detach().cpu().tolist(),
        "timing": (
            "raw_per_iteration_cuda_event"
            if use_cuda
            else "raw_per_iteration_perf_counter"
        ),
        "timing_unit": "seconds",
        "timing_is_amortized": False,
        "warmup_iterations": int(bool(warmup)),
        "tf32": False,
        "tol": 0.0,
    }
    if use_cuda:
        metrics.update(
            peak_allocated_bytes=int(
                torch.cuda.max_memory_allocated(device=resolved_device)
            ),
            peak_reserved_bytes=int(
                torch.cuda.max_memory_reserved(device=resolved_device)
            ),
        )
    return metrics


def _torchnmf_exact_zero_update(
    torch: Any,
    V: Any,
    WH: Any,
    param: Any,
    beta: float,
    gamma: float,
    l1_reg: float,
    l2_reg: float,
    pos: Any = None,
) -> None:
    """torchnmf update adapted to sklearn's exact-zero denominator rule."""
    if beta != 2:
        raise ValueError("the controlled torchnmf benchmark requires beta=2")
    param.grad = None
    WH.backward(V, retain_graph=pos is None)
    numerator = param.grad.relu_().clone()

    if pos is None:
        param.grad = None
        WH.backward(WH)
        denominator = param.grad.relu_().clone()
    else:
        denominator = pos.clone()

    if l1_reg > 0:
        denominator.add_(l1_reg)
    if l2_reg > 0:
        denominator.add_(param.data, alpha=l2_reg)
    eps = torch.tensor(
        SKLEARN_EXACT_ZERO_EPSILON,
        dtype=denominator.dtype,
        device=denominator.device,
    )
    denominator = denominator.where(denominator != 0, eps)
    multiplier = numerator.div_(denominator)
    if gamma != 1:
        multiplier.pow_(gamma)
    param.data.mul_(multiplier)


@contextlib.contextmanager
def _patched_torchnmf_zero_guard(torch: Any):
    try:
        import torchnmf.nmf as torchnmf_module
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torchnmf is required for the torch_nmf benchmark backend"
        ) from exc
    original = torchnmf_module._double_backward_update
    torchnmf_module._double_backward_update = (
        lambda V, WH, param, beta, gamma, l1_reg, l2_reg, pos=None: (
            _torchnmf_exact_zero_update(
                torch, V, WH, param, beta, gamma, l1_reg, l2_reg, pos
            )
        )
    )
    try:
        yield torchnmf_module
    finally:
        torchnmf_module._double_backward_update = original


def run_torchnmf_checkpoints(
    X: np.ndarray,
    W0: np.ndarray,
    H0: np.ndarray,
    *,
    checkpoints: Sequence[int],
    callback: CheckpointCallback,
    device: str = "cuda",
    dtype: str = "fp32",
) -> dict[str, float | str | int]:
    """Run torchnmf from explicit factors in sklearn's W-then-H update order.

    ``torchnmf.BaseComponent.fit`` updates its template ``W`` before its
    activation ``H``. With the shape mapping required for ``X ~= W @ H``,
    that is the reverse of sklearn's update order. Drive torchnmf's own
    autograd updater explicitly here so the controlled MU comparison first
    updates canonical W (usages), then canonical H (spectra).
    """
    checkpoints = _validated_checkpoints(checkpoints)
    X, W, H = _validated_factors(X, W0, H0, batched=False)
    torch = _loud_import_torch()
    resolved_device = _select_device(torch, device)
    resolved_dtype = _select_storage(torch, dtype, resolved_device)
    expected_dtype = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
    }.get(X.dtype)
    if expected_dtype is None or resolved_dtype is not expected_dtype:
        raise ValueError(
            f"requested Torch dtype {resolved_dtype} does not match X dtype {X.dtype}"
        )
    torch.use_deterministic_algorithms(True)

    V = torch.as_tensor(X, dtype=resolved_dtype, device=resolved_device)
    with _patched_torchnmf_zero_guard(torch) as torchnmf_module:
        # torchnmf uses V ~= model.H @ model.W.T.
        model = torchnmf_module.NMF(
            rank=H.shape[0],
            W=torch.as_tensor(H.T.copy(), dtype=resolved_dtype),
            H=torch.as_tensor(W.copy(), dtype=resolved_dtype),
        ).to(device=resolved_device, dtype=resolved_dtype)
        # torchnmf 0.3.5 allocates Parameter storage with Torch's default
        # dtype inside its constructor, even when the supplied tensors are
        # fp64. Re-copy after converting the module so fp64 initialization is
        # not silently rounded through fp32.
        with torch.no_grad():
            model.W.copy_(
                torch.as_tensor(
                    H.T,
                    dtype=resolved_dtype,
                    device=resolved_device,
                )
            )
            model.H.copy_(
                torch.as_tensor(
                    W,
                    dtype=resolved_dtype,
                    device=resolved_device,
                )
            )

        wanted = set(checkpoints)
        for iteration in range(1, checkpoints[-1] + 1):
            # Update canonical W == model.H from the old canonical H.
            reconstruction = model.reconstruct(model.H, model.W.detach())
            torchnmf_module._double_backward_update(
                V,
                reconstruction,
                model.H,
                2,
                1.0,
                0.0,
                0.0,
            )

            # Update canonical H == model.W.T from the new canonical W.
            reconstruction = model.reconstruct(model.H.detach(), model.W)
            torchnmf_module._double_backward_update(
                V,
                reconstruction,
                model.W,
                2,
                1.0,
                0.0,
                0.0,
            )

            if iteration in wanted:
                if resolved_device.startswith("cuda"):
                    torch.cuda.synchronize()
                callback(
                    iteration,
                    model.H.detach().cpu().numpy(),
                    model.W.detach().T.cpu().numpy(),
                )

    return {
        "device": resolved_device,
        "dtype": dtype,
        "batch_size": 1,
        "zero_guard": "replace_exact_zero_with_float32_eps",
        "epsilon": SKLEARN_EXACT_ZERO_EPSILON,
    }

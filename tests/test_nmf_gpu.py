"""Reliability tests for the NMF GPU kernel (`src/cnmf/nmf_gpu.py`).

Scope
-----
These tests cover the kernel as a standalone NMF implementation. They do not
exercise Nextflow wiring or cNMF process orchestration. The default path is CPU
so the suite can run in ordinary CI; CUDA checks are present but skipped when a
CUDA device is unavailable.

Sections
--------
Public API and reconstruction contract
    Verifies that factorization actually reduces reconstruction error, returns
    the cNMF-compatible `(spectra, usages) = (H, W)` order, keeps float64 numpy
    outputs for compatibility, and preserves the thin `nmf_gpu` adapter shape.

MU update order, convergence, and iteration bounds
    Pins the sklearn-style W-then-H multiplicative-update order, early-stop
    behavior, exact `max_iter` cap handling, and same-dtype runtime guard.

torch.compile behavior
    Checks that compiled execution is equivalent to eager execution for the same
    seed/options, and that explicit `compile_block` is the cadence used when
    compile is enabled.

Initialization and reproducibility
    Covers random-state determinism/diversity, `init=None -> random`, sklearn
    initializer parity, nndsvd-family pass-through, and unsupported custom init.

Input validation and degenerate shapes
    Exercises negative/NaN/inf rejection, zero matrices, one-row/one-column
    inputs, empty or non-2D input, zero rank, and `k > min(cells, genes)`.

Runtime option parsing
    Ensures all GPU options come from `gpu_kwargs`, defaults are centralized,
    string booleans/numerics are parsed, and iteration cadences are at least one.

Device, dtype, imports, sparse, and backend policy
    Uses fake backends for portable device/dtype policy checks, verifies loud
    dependency errors, covers the current sparse densify path, and keeps
    CUDA-only fp32/bf16/TF32 behavior behind device-gated tests.

sklearn MU parity
    Compares final aligned W/H factors against sklearn Frobenius MU for fp64
    and fp32 with identical initialization, seed, and iteration count. Small
    cases run routinely on the torch CPU backend and CUDA when available. An
    opt-in, memory-gated CUDA stress case uses a 100,000 x 20,000 matrix.
"""

import argparse
import builtins
import gc
import os
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from utils import (
    assert_valid_nmf_output,
    fake_torch_backend,
    kernel,
    load_kernel_module,
    low_rank_matrix,
    require_nmf_runtime,
    small_nonnegative_matrix,
)


# ---------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------
def test_kernel_loader_fails_when_kernel_file_is_missing(tmp_path):
    """Fail the test harness clearly if the kernel module is absent."""
    missing_kernel = tmp_path / "missing_nmf_gpu.py"

    with pytest.raises(pytest.fail.Exception, match="Required NMF GPU kernel file is missing"):
        load_kernel_module(module_name="missing_nmf_gpu_for_test", kernel_path=missing_kernel)


# ---------------------------------------------------------------------
# Public API and reconstruction contract
# ---------------------------------------------------------------------
def test_nmf_gpu_reconstructs_known_low_rank_matrix_with_small_relative_error(kernel):
    """Factorize an exact low-rank non-negative matrix and require low relative error."""
    require_nmf_runtime()
    k = 3
    X = low_rank_matrix(rank=k)

    H, W = kernel._nmf_gpu(
        X,
        {"n_components": k, "max_iter": 600, "tol": 0, "random_state": 0},
        {"device": "cpu", "check_every": 600},
    )

    assert_valid_nmf_output(X, H, W, k)
    rel = np.linalg.norm(X - W @ H) / np.linalg.norm(X)
    assert rel < 1e-3


def test_nmf_gpu_returns_spectra_then_usages_with_cnmf_orientation(kernel):
    """Pin the public return order as spectra H first, usages W second."""
    require_nmf_runtime()
    X = small_nonnegative_matrix(cells=7, genes=5)

    H, W = kernel._nmf_gpu(
        X,
        {"n_components": 2, "max_iter": 2, "random_state": 0},
        {"device": "cpu"},
    )

    assert H.shape == (2, 5)
    assert W.shape == (7, 2)


def test_nmf_gpu_cpu_smoke_shapes_dtype_sign_and_finiteness(kernel):
    """Smoke-test CPU output shape, float64 compatibility dtype, finite values, and non-negativity."""
    require_nmf_runtime()
    X = small_nonnegative_matrix()
    H, W = kernel._nmf_gpu(
        X,
        {"n_components": 3, "max_iter": 3, "random_state": 0},
        {"device": "cpu"},
    )

    assert_valid_nmf_output(X, H, W, 3)


def test_nmf_gpu_fp32_compute_still_returns_float64_numpy_outputs(kernel):
    """Exercise fp32 compute while keeping the public numpy output contract as float64."""
    require_nmf_runtime()
    X = small_nonnegative_matrix()

    H, W = kernel._nmf_gpu(
        X,
        {"n_components": 2, "max_iter": 1, "random_state": 0},
        {"device": "cpu", "dtype": "fp32"},
    )

    assert H.dtype == np.float64
    assert W.dtype == np.float64


def test_nmf_gpu_adapter_ignores_self_and_delegates_to_nmf_gpu(kernel, monkeypatch):
    """Verify the cNMF adapter extracts embedded GPU args and ignores its bound `self`."""
    calls = []
    sentinel = (object(), object())

    def fake_factorize(X, nmf_kwargs, gpu_kwargs=None):
        calls.append((X, nmf_kwargs, gpu_kwargs))
        return sentinel

    monkeypatch.setattr(kernel, "_nmf_gpu", fake_factorize)
    X = np.ones((3, 2))
    gpu_kwargs = {"device": "cpu"}
    nmf_kwargs = {"engine": "gpu", "gpu": gpu_kwargs, "n_components": 1}

    result = kernel.nmf_gpu(object(), X, nmf_kwargs)

    assert result is sentinel
    assert calls == [(X, {"n_components": 1}, gpu_kwargs)]
    assert nmf_kwargs == {"engine": "gpu", "gpu": gpu_kwargs, "n_components": 1}


# ---------------------------------------------------------------------
# MU update order, convergence, and iteration bounds
# ---------------------------------------------------------------------
def test_mu_step_updates_w_first_using_old_h_then_h_using_new_w(kernel):
    """Check one MU step matches sklearn parity: update W from old H, then H from new W."""
    torch = require_nmf_runtime()
    Xg = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    W0 = torch.tensor([[0.5, 0.7], [0.9, 1.1]], dtype=torch.float64)
    H0 = torch.tensor([[0.6, 0.8], [1.0, 1.2]], dtype=torch.float64)
    eps = torch.tensor(1e-9, dtype=torch.float64)

    denominator = W0 @ (H0 @ H0.T)
    denominator = denominator.where(denominator != 0, eps)
    expected_W = W0 * ((Xg @ H0.T) / denominator)
    denominator = (expected_W.T @ expected_W) @ H0
    denominator = denominator.where(denominator != 0, eps)
    expected_H = H0 * ((expected_W.T @ Xg) / denominator)
    W, H = kernel._mu_step(W0, H0, Xg, eps)

    assert torch.allclose(W, expected_W)
    assert torch.allclose(H, expected_H)


def test_mu_step_fixed_h_matches_sklearn_exact_zero_protection(kernel):
    """Replace exact zeros while leaving tiny nonzero denominators untouched."""
    torch = require_nmf_runtime()
    eps = torch.tensor(np.finfo(np.float32).eps, dtype=torch.float64)

    zero_result = kernel._mu_step_fixed_h(
        torch.ones((1, 1), dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64),
        eps,
    )
    assert torch.equal(zero_result, torch.zeros_like(zero_result))
    assert torch.isfinite(zero_result).all()

    tiny = torch.tensor(1e-12, dtype=torch.float64)
    tiny_result = kernel._mu_step_fixed_h(
        torch.ones((1, 1), dtype=torch.float64),
        tiny.reshape(1, 1),
        torch.ones((1, 1), dtype=torch.float64),
        eps,
    )
    assert torch.equal(tiny_result, (1 / tiny).reshape(1, 1))


def test_fit_mu_early_stops_when_relative_error_drop_is_below_tol(kernel):
    """Confirm the MU loop stops after a flat relative-error drop crosses the tolerance."""
    torch = require_nmf_runtime()
    Xg = torch.full((3, 2), 2.0, dtype=torch.float64)
    W = torch.ones((3, 1), dtype=torch.float64)
    H = torch.ones((1, 2), dtype=torch.float64)
    eps = torch.tensor(1e-9, dtype=torch.float64)
    calls = {"count": 0}

    def no_change_step(W, H, Xg, eps):
        calls["count"] += 1
        return W, H

    kernel._fit_mu(torch, Xg, W, H, eps, 10, 1e-4, no_change_step, 1, False, "cpu")

    assert calls["count"] == 2


def test_fit_mu_respects_max_iter_without_overrunning_final_block(kernel):
    """Ensure block execution clips the last block instead of running past max_iter."""
    torch = require_nmf_runtime()
    Xg = torch.full((3, 2), 2.0, dtype=torch.float64)
    W = torch.ones((3, 1), dtype=torch.float64)
    H = torch.ones((1, 2), dtype=torch.float64)
    eps = torch.tensor(1e-9, dtype=torch.float64)
    calls = {"count": 0}

    def no_change_step(W, H, Xg, eps):
        calls["count"] += 1
        return W, H

    kernel._fit_mu(torch, Xg, W, H, eps, 6, -1.0, no_change_step, 4, False, "cpu")

    assert calls["count"] == 6


def test_check_runtime_tensors_rejects_mixed_dtypes(kernel):
    """Reject mixed runtime tensor dtypes so storage precision is also matmul precision."""
    torch = require_nmf_runtime()
    Xg = torch.ones((2, 2), dtype=torch.float32)
    W = torch.ones((2, 1), dtype=torch.float32)
    H = torch.ones((1, 2), dtype=torch.float64)
    eps = torch.tensor(1e-9, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="share dtype"):
        kernel._check_runtime_tensors(Xg, W, H, eps)


# ---------------------------------------------------------------------
# torch.compile behavior
# ---------------------------------------------------------------------
def test_compile_mode_matches_eager_output_for_same_seed_and_options(kernel, monkeypatch):
    """Use a compile stub to require compiled and eager paths to produce identical factors."""
    torch = require_nmf_runtime()
    monkeypatch.setattr(torch, "compile", lambda fn: fn)
    X = small_nonnegative_matrix(cells=8, genes=6)
    nmf_kwargs = {"n_components": 2, "max_iter": 4, "tol": -1.0, "random_state": 0}

    eager_H, eager_W = kernel._nmf_gpu(
        X,
        nmf_kwargs,
        {"device": "cpu", "dtype": "fp64", "compile": False, "check_every": 1},
    )
    compiled_H, compiled_W = kernel._nmf_gpu(
        X,
        nmf_kwargs,
        {"device": "cpu", "dtype": "fp64", "compile": True, "compile_block": 2},
    )

    assert np.allclose(compiled_H, eager_H)
    assert np.allclose(compiled_W, eager_W)


def test_compile_mode_uses_explicit_multi_iteration_compile_block_when_requested(kernel, monkeypatch):
    """Pin explicit compile_block as the convergence-check cadence for compiled execution."""
    torch = require_nmf_runtime()
    calls = []
    monkeypatch.setattr(torch, "compile", lambda fn: calls.append(fn) or fn)
    opt = dict(kernel.DEFAULT_GPU, compile=True, check_every=1, compile_block=3)

    step, block = kernel._execution_plan(torch, opt, "cpu")

    assert calls == [kernel._mu_step]
    assert step is kernel._mu_step
    assert block == 3


# ---------------------------------------------------------------------
# Initialization and reproducibility
# ---------------------------------------------------------------------
def test_nmf_gpu_random_state_is_reproducible(kernel):
    """The same random_state should produce identical initialization and final factors."""
    require_nmf_runtime()
    X = small_nonnegative_matrix()
    kwargs = {"n_components": 3, "max_iter": 3, "random_state": 13}
    gpu = {"device": "cpu", "check_every": 3}

    H1, W1 = kernel._nmf_gpu(X, kwargs, gpu)
    H2, W2 = kernel._nmf_gpu(X, kwargs, gpu)

    assert np.allclose(H1, H2)
    assert np.allclose(W1, W2)


def test_nmf_gpu_different_random_state_changes_result(kernel):
    """Different random_state values should produce different random initial factors."""
    require_nmf_runtime()
    X = small_nonnegative_matrix()
    kwargs = {"n_components": 3, "max_iter": 0, "init": "random"}

    H1, W1 = kernel._nmf_gpu(X, dict(kwargs, random_state=1), {"device": "cpu"})
    H2, W2 = kernel._nmf_gpu(X, dict(kwargs, random_state=2), {"device": "cpu"})

    assert not np.allclose(H1, H2)
    assert not np.allclose(W1, W2)


def test_init_none_defaults_to_random_init(kernel, monkeypatch):
    """Preserve cNMF consensus behavior by mapping init=None to random initialization."""
    seen = []

    def fake_initialize(X, n_components, init, random_state):
        seen.append(init)
        return np.ones((X.shape[0], n_components)), np.ones((n_components, X.shape[1]))

    monkeypatch.setattr(kernel, "_loud_import_initialize_nmf", lambda: fake_initialize)

    kernel._init_wh(small_nonnegative_matrix(), 2, 0, None)

    assert seen == ["random"]


def test_random_init_matches_sklearn_initializer_contract(kernel):
    """Compare random initialization directly against sklearn's private initializer."""
    pytest.importorskip("sklearn")
    from sklearn.decomposition._nmf import _initialize_nmf

    X = small_nonnegative_matrix()
    expected_W, expected_H = _initialize_nmf(X, n_components=3, init="random", random_state=5)
    W, H = kernel._init_wh(X, 3, 5, "random")

    assert np.allclose(W, expected_W)
    assert np.allclose(H, expected_H)


def test_nndsvd_initializers_pass_through_to_sklearn_initializer(kernel, monkeypatch):
    """Ensure nndsvd, nndsvda, and nndsvdar are forwarded unchanged to sklearn."""
    seen = []

    def fake_initialize(X, n_components, init, random_state):
        seen.append(init)
        return np.ones((X.shape[0], n_components)), np.ones((n_components, X.shape[1]))

    monkeypatch.setattr(kernel, "_loud_import_initialize_nmf", lambda: fake_initialize)

    for init in ("nndsvd", "nndsvda", "nndsvdar"):
        kernel._init_wh(small_nonnegative_matrix(), 2, 0, init)

    assert seen == ["nndsvd", "nndsvda", "nndsvdar"]


def test_nmf_gpu_custom_init_raises(kernel):
    """Document that custom W/H initialization is not implemented in this standalone API."""
    with pytest.raises(NotImplementedError, match="custom"):
        kernel._init_wh(small_nonnegative_matrix(), 2, 0, "custom")


# ---------------------------------------------------------------------
# sklearn Frobenius-MU parity: same input, initialization, seed, iterations
# ---------------------------------------------------------------------
PARITY_CASES = [
    pytest.param("fp64", np.float64, 2e-8, 1e-9, id="fp64"),
    pytest.param("fp32", np.float32, 2e-5, 2e-6, id="fp32"),
]

LARGE_PARITY_ENV = "CNMF_RUN_LARGE_GPU_PARITY"
LARGE_PARITY_ROWS = 100_000
LARGE_PARITY_COLUMNS = 20_000
LARGE_PARITY_COMPONENTS = 2


def _parity_nmf_kwargs(n_components, seed, max_iter):
    """Return the common, unregularized sklearn/GPU Frobenius-MU contract."""
    return {
        "n_components": n_components,
        "init": "random",
        "random_state": seed,
        "solver": "mu",
        "beta_loss": "frobenius",
        "tol": 0.0,
        "max_iter": max_iter,
        "alpha_W": 0.0,
        "alpha_H": 0.0,
        "l1_ratio": 0.0,
    }


def _sklearn_mu_reference(X, nmf_kwargs):
    """Run sklearn MU and require the requested fixed iteration count."""
    pytest.importorskip("sklearn")
    from sklearn.decomposition import non_negative_factorization
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        W, H, n_iter = non_negative_factorization(X, **nmf_kwargs)
    assert n_iter == nmf_kwargs["max_iter"]
    return H, W


def _gpu_parity_result(kernel, X, nmf_kwargs, dtype_name, device):
    """Run the PyTorch kernel without TF32, compilation, or early stopping."""
    return kernel._nmf_gpu(
        X,
        nmf_kwargs,
        {
            "device": device,
            "dtype": dtype_name,
            "allow_tf32": False,
            "compile": False,
            # One convergence block means every requested MU iteration runs
            # before the first possible tolerance check.
            "check_every": nmf_kwargs["max_iter"],
        },
    )


def _relative_reconstruction_error(X, H, W):
    """Compute reconstruction error in float64, independent of compute dtype."""
    X64 = np.asarray(X, dtype=np.float64)
    H64 = np.asarray(H, dtype=np.float64)
    W64 = np.asarray(W, dtype=np.float64)
    return np.linalg.norm(X64 - W64 @ H64) / np.linalg.norm(X64)


@pytest.mark.parametrize("dtype_name,np_dtype,rtol,atol", PARITY_CASES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("seed", [0, 17, 101])
@pytest.mark.parametrize("max_iter", [1, 25])
def test_sklearn_mu_matches_gpu_kernel_on_small_matrix(
    kernel, dtype_name, np_dtype, rtol, atol, device, seed, max_iter
):
    """Match sklearn exactly on CPU and numerically on CUDA across aligned seeds."""
    torch = require_nmf_runtime()
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    X = np.random.default_rng(42).random((64, 48), dtype=np_dtype)
    X += np_dtype(0.1)  # strictly positive denominators avoid degenerate MU behavior
    nmf_kwargs = _parity_nmf_kwargs(
        n_components=4,
        seed=seed,
        max_iter=max_iter,
    )

    expected_H, expected_W = _sklearn_mu_reference(X, nmf_kwargs)
    actual_H, actual_W = _gpu_parity_result(
        kernel, X, nmf_kwargs, dtype_name, device
    )

    if device == "cpu":
        np.testing.assert_array_equal(actual_H, expected_H)
        np.testing.assert_array_equal(actual_W, expected_W)
    else:
        np.testing.assert_allclose(actual_H, expected_H, rtol=rtol, atol=atol)
        np.testing.assert_allclose(actual_W, expected_W, rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            _relative_reconstruction_error(X, actual_H, actual_W),
            _relative_reconstruction_error(X, expected_H, expected_W),
            rtol=rtol,
            atol=atol,
        )


def _available_host_memory_bytes():
    """Return Linux available host memory when sysconf exposes it."""
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def _require_large_cuda_parity_capacity(torch, np_dtype):
    """Skip the destructive-size stress case unless explicitly enabled and safe."""
    if os.environ.get(LARGE_PARITY_ENV) != "1":
        pytest.skip(f"set {LARGE_PARITY_ENV}=1 to run the 100K x 20K parity stress test")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    x_bytes = (
        LARGE_PARITY_ROWS
        * LARGE_PARITY_COLUMNS
        * np.dtype(np_dtype).itemsize
    )
    available_host = _available_host_memory_bytes()
    required_host = 3 * x_bytes
    if available_host is not None and available_host < required_host:
        pytest.skip(
            f"large parity needs about {required_host / 2**30:.1f} GiB available host "
            f"memory; found {available_host / 2**30:.1f} GiB"
        )

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    required_device = int(1.25 * x_bytes)
    if properties.total_memory < required_device:
        pytest.skip(
            f"large parity needs a device with at least {required_device / 2**30:.1f} "
            f"GiB addressable memory; found {properties.total_memory / 2**30:.1f} GiB"
        )
    if not getattr(properties, "is_integrated", False):
        free_device, _ = torch.cuda.mem_get_info()
        if free_device < required_device:
            pytest.skip(
                f"large parity needs about {required_device / 2**30:.1f} GiB free GPU "
                f"memory; found {free_device / 2**30:.1f} GiB"
            )


@pytest.mark.parametrize("dtype_name,np_dtype,rtol,atol", PARITY_CASES)
def test_sklearn_mu_matches_cuda_on_100k_by_20k_matrix(
    kernel, dtype_name, np_dtype, rtol, atol
):
    """Stress sklearn/CUDA parity on a 100K x 20K dense matrix for one MU iteration."""
    torch = require_nmf_runtime()
    _require_large_cuda_parity_capacity(torch, np_dtype)

    X = np.random.default_rng(42).random(
        (LARGE_PARITY_ROWS, LARGE_PARITY_COLUMNS),
        dtype=np_dtype,
    )
    X += np_dtype(0.1)
    nmf_kwargs = _parity_nmf_kwargs(
        n_components=LARGE_PARITY_COMPONENTS,
        seed=17,
        max_iter=1,
    )

    try:
        expected_H, expected_W = _sklearn_mu_reference(X, nmf_kwargs)
        actual_H, actual_W = _gpu_parity_result(
            kernel, X, nmf_kwargs, dtype_name, "cuda"
        )

        np.testing.assert_allclose(actual_H, expected_H, rtol=rtol, atol=atol)
        np.testing.assert_allclose(actual_W, expected_W, rtol=rtol, atol=atol)
    finally:
        # Do not make a following dtype inherit this case's multi-GiB CUDA cache.
        del X
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Input validation and degenerate shapes
# ---------------------------------------------------------------------
def test_nmf_gpu_rejects_negative_input(kernel):
    """NMF input must be non-negative."""
    with pytest.raises(ValueError, match="non-negative"):
        kernel._to_checked_array(np.array([[1.0, -0.1]]))


def test_nmf_gpu_rejects_nan_input(kernel):
    """NaN input should fail before torch/sklearn runtime work begins."""
    with pytest.raises(ValueError, match="NaN/inf"):
        kernel._to_checked_array(np.array([[1.0, np.nan]]))


def test_nmf_gpu_rejects_inf_input(kernel):
    """Infinite input should fail before torch/sklearn runtime work begins."""
    with pytest.raises(ValueError, match="NaN/inf"):
        kernel._to_checked_array(np.array([[1.0, np.inf]]))


def test_nmf_gpu_zero_matrix_does_not_crash_or_divide_by_zero(kernel):
    """All-zero input should take the degenerate path without crashing or producing invalid output."""
    require_nmf_runtime()
    X = np.zeros((5, 4))

    H, W = kernel._nmf_gpu(
        X,
        {"n_components": 2, "max_iter": 5, "random_state": 0},
        {"device": "cpu"},
    )

    assert_valid_nmf_output(X, H, W, 2)


def test_nmf_gpu_handles_single_row_and_single_column_inputs(kernel):
    """Single-row and single-column matrices should keep valid H/W orientation."""
    require_nmf_runtime()

    H_row, W_row = kernel._nmf_gpu(
        np.array([[1.0, 2.0, 3.0]]),
        {"n_components": 1, "max_iter": 1, "random_state": 0},
        {"device": "cpu"},
    )
    H_col, W_col = kernel._nmf_gpu(
        np.array([[1.0], [2.0], [3.0]]),
        {"n_components": 1, "max_iter": 1, "random_state": 0},
        {"device": "cpu"},
    )

    assert H_row.shape == (1, 3)
    assert W_row.shape == (1, 1)
    assert H_col.shape == (1, 1)
    assert W_col.shape == (3, 1)


def test_nmf_gpu_rejects_empty_or_zero_dimensional_inputs(kernel):
    """Reject empty matrices and non-2D arrays with clear validation errors."""
    for X in (np.empty((0, 3)), np.empty((3, 0))):
        with pytest.raises(ValueError, match="at least one row"):
            kernel._to_checked_array(X)

    with pytest.raises(ValueError, match="2D matrix"):
        kernel._to_checked_array(np.array([1.0, 2.0]))


def test_nmf_gpu_rejects_zero_components(kernel):
    """Reject rank k=0 before reaching sklearn's initializer."""
    require_nmf_runtime()
    with pytest.raises(ValueError, match="n_components"):
        kernel._nmf_gpu(
            np.ones((3, 3)),
            {"n_components": 0, "max_iter": 1, "random_state": 0},
            {"device": "cpu"},
        )


def test_nmf_gpu_defines_behavior_when_k_exceeds_min_dimension(kernel):
    """Allow sklearn-compatible overcomplete factorization when k exceeds matrix dimensions."""
    require_nmf_runtime()
    X = small_nonnegative_matrix(cells=3, genes=2)

    H, W = kernel._nmf_gpu(
        X,
        {"n_components": 4, "max_iter": 1, "random_state": 0},
        {"device": "cpu"},
    )

    assert_valid_nmf_output(X, H, W, 4)


# ---------------------------------------------------------------------
# Runtime option parsing
# ---------------------------------------------------------------------
def test_resolve_gpu_opts_uses_defaults_when_gpu_kwargs_is_missing(kernel):
    """Missing gpu_kwargs should resolve exactly to the centralized DEFAULT_GPU values."""
    assert kernel._resolve_gpu_opts(None) == kernel.DEFAULT_GPU


def test_default_gpu_epsilon_matches_sklearn_exact_value(kernel):
    """Pin sklearn's float32 epsilon without importing its private EPSILON symbol."""
    assert kernel.DEFAULT_GPU["eps"] == float(np.finfo(np.float32).eps)


def test_resolve_gpu_opts_dict_values_override_defaults(kernel):
    """Explicit gpu_kwargs values override defaults and are normalized to typed options."""
    opts = kernel._resolve_gpu_opts(
        {
            "device": "CUDA:1",
            "dtype": "FP32",
            "allow_tf32": "yes",
            "compile": "on",
            "eps": "1e-8",
            "check_every": "7",
            "compile_block": "9",
        }
    )

    assert opts == {
        "device": "cuda:1",
        "dtype": "fp32",
        "allow_tf32": True,
        "compile": True,
        "eps": 1e-8,
        "check_every": 7,
        "compile_block": 9,
        "batch": 1,             # not overridden here -> default
    }


def test_resolve_gpu_opts_reads_only_gpu_kwargs_not_environment_variables(kernel, monkeypatch):
    """Environment variables should not affect option resolution for this kernel."""
    monkeypatch.setenv("CNMF_GPU_DTYPE", "bf16")
    monkeypatch.setenv("CNMF_GPU_COMPILE", "true")

    opts = kernel._resolve_gpu_opts({})

    assert opts["dtype"] == kernel.DEFAULT_GPU["dtype"]
    assert opts["compile"] is kernel.DEFAULT_GPU["compile"]


def test_resolve_gpu_opts_parses_truthy_boolean_strings(kernel):
    """Truthy strings accepted by Nextflow config should become real booleans."""
    for value in ("1", "true", "TRUE", "yes", "on", True):
        opts = kernel._resolve_gpu_opts({"allow_tf32": value, "compile": value})
        assert opts["allow_tf32"] is True
        assert opts["compile"] is True


def test_resolve_gpu_opts_parses_false_for_non_truthy_boolean_strings(kernel):
    """Non-truthy boolean strings should resolve to False."""
    for value in ("0", "false", "off", "no", "", False):
        opts = kernel._resolve_gpu_opts({"allow_tf32": value, "compile": value})
        assert opts["allow_tf32"] is False
        assert opts["compile"] is False


def test_resolve_gpu_opts_coerces_numeric_strings_to_float_and_int(kernel):
    """Numeric config strings should be coerced to the expected float/int types."""
    opts = kernel._resolve_gpu_opts({"eps": "0.125", "check_every": "4", "compile_block": "5"})

    assert opts["eps"] == 0.125
    assert opts["check_every"] == 4
    assert opts["compile_block"] == 5


def test_resolve_gpu_opts_floors_check_every_and_compile_block_to_at_least_one(kernel):
    """Iteration cadence options should never resolve below one."""
    opts = kernel._resolve_gpu_opts({"check_every": 0, "compile_block": -3})

    assert opts["check_every"] == 1
    assert opts["compile_block"] == 1


# ---------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------
def test_select_device_auto_prefers_cuda_then_mps_then_cpu(kernel):
    """Auto device selection should prefer CUDA, then MPS, then CPU."""
    assert kernel._select_device(fake_torch_backend(cuda_available=True, mps_available=True), "auto") == "cuda"
    assert kernel._select_device(fake_torch_backend(cuda_available=False, mps_available=True), "auto") == "mps"
    assert kernel._select_device(fake_torch_backend(cuda_available=False, mps_available=False), "auto") == "cpu"


def test_select_device_invalid_device_raises(kernel):
    """Unknown device names should fail loudly instead of falling back."""
    with pytest.raises(ValueError, match="not recognized"):
        kernel._select_device(fake_torch_backend(), "gpu")


def test_select_device_explicit_unavailable_cuda_or_mps_raises(kernel):
    """Explicit unavailable GPU devices should raise rather than silently using CPU."""
    fake = fake_torch_backend(cuda_available=False, mps_available=False)

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        kernel._select_device(fake, "cuda")
    with pytest.raises(RuntimeError, match="MPS is unavailable"):
        kernel._select_device(fake, "mps")


# ---------------------------------------------------------------------
# Dtype and storage selection
# ---------------------------------------------------------------------
def test_select_storage_auto_cpu_is_fp64(kernel):
    """Auto dtype on CPU should select fp64 for a stable reference path."""
    assert kernel._select_storage(fake_torch_backend(), "auto", "cpu") == "float64"


def test_select_storage_auto_gpu_is_fp32(kernel):
    """Auto dtype on GPU-class backends should select fp32."""
    fake = fake_torch_backend()
    assert kernel._select_storage(fake, "auto", "cuda:0") == "float32"
    assert kernel._select_storage(fake, "auto", "mps") == "float32"


def test_select_storage_invalid_dtype_raises(kernel):
    """Unknown dtype names should fail with a clear configuration error."""
    with pytest.raises(ValueError, match="not recognized"):
        kernel._select_storage(fake_torch_backend(), "fp16", "cpu")


def test_select_storage_fp64_on_mps_raises(kernel):
    """MPS should reject explicit fp64 because this kernel treats MPS as fp32-only."""
    with pytest.raises(RuntimeError, match="MPS has no fp64"):
        kernel._select_storage(fake_torch_backend(), "fp64", "mps")


def test_select_storage_bf16_is_cuda_only(kernel):
    """bf16 is accepted only for CUDA and means bf16 storage plus bf16 matmul operands."""
    with pytest.raises(RuntimeError, match="only supported on CUDA"):
        kernel._select_storage(fake_torch_backend(), "bf16", "cpu")

    assert kernel._select_storage(fake_torch_backend(bf16_supported=True), "bf16", "cuda") == "bfloat16"


def test_select_storage_bf16_checks_cuda_device_support(kernel):
    """CUDA bf16 requests should check the actual device capability."""
    with pytest.raises(RuntimeError, match="does not support bf16"):
        kernel._select_storage(fake_torch_backend(bf16_supported=False), "bf16", "cuda")


# ---------------------------------------------------------------------
# Loud dependency imports
# ---------------------------------------------------------------------
def test_loud_import_torch_missing_has_actionable_error(kernel, monkeypatch):
    """Missing torch should raise an actionable environment setup message."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="PyTorch is required"):
        kernel._loud_import_torch()


def test_loud_import_sklearn_missing_has_actionable_error(kernel, monkeypatch):
    """Missing sklearn should raise an actionable environment setup message."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sklearn.decomposition._nmf":
            raise ModuleNotFoundError("No module named 'sklearn'", name="sklearn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="scikit-learn is required"):
        kernel._loud_import_initialize_nmf()


def test_loud_import_sklearn_incompatible_initializer_has_actionable_error(kernel, monkeypatch):
    """A sklearn version without _initialize_nmf should fail with a version-focused message."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sklearn.decomposition._nmf":
            raise ImportError("cannot import name '_initialize_nmf'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="does not expose"):
        kernel._loud_import_initialize_nmf()


# ---------------------------------------------------------------------
# Sparse input and initializer variants
# ---------------------------------------------------------------------
def test_sparse_input_uses_densify_path_and_returns_valid_output(kernel):
    """Current sparse support densifies input and still returns valid factors."""
    require_nmf_runtime()
    sparse = pytest.importorskip("scipy.sparse")
    X = sparse.csr_matrix(small_nonnegative_matrix(cells=6, genes=5))

    H, W = kernel._nmf_gpu(
        X,
        {"n_components": 2, "max_iter": 2, "random_state": 0},
        {"device": "cpu"},
    )

    assert_valid_nmf_output(X.toarray(), H, W, 2)


def test_nndsvd_nndsvda_nndsvdar_initializers_return_valid_outputs(kernel):
    """Each supported nndsvd-family initializer should run end-to-end."""
    require_nmf_runtime()
    X = small_nonnegative_matrix(cells=8, genes=6)

    for init in ("nndsvd", "nndsvda", "nndsvdar"):
        H, W = kernel._nmf_gpu(
            X,
            {"n_components": 3, "max_iter": 1, "random_state": 0, "init": init},
            {"device": "cpu"},
        )
        assert_valid_nmf_output(X, H, W, 3)


# ---------------------------------------------------------------------
# Backend-specific execution behavior
# ---------------------------------------------------------------------
def test_execution_plan_ignores_compile_on_mps_and_uses_eager_path(kernel):
    """MPS compile requests should resolve to eager execution with check_every cadence."""
    fake_torch = SimpleNamespace(compile=lambda fn: pytest.fail("compile should be ignored on MPS"))
    opt = dict(kernel.DEFAULT_GPU, compile=True, check_every=4, compile_block=9)

    step, block = kernel._execution_plan(fake_torch, opt, "mps")

    assert step is kernel._mu_step
    assert block == 4


def test_tf32_flag_is_applied_only_for_cuda_float32_compute(kernel, monkeypatch):
    """allow_tf32 should not be passed into the fit loop for non-CUDA execution."""
    require_nmf_runtime()
    captured = []

    def fake_fit_mu(torch, Xg, W, H, eps, max_iter, tol, step, block, tf32, device):
        captured.append((tf32, device, Xg.dtype))
        return W, H

    monkeypatch.setattr(kernel, "_fit_mu", fake_fit_mu)
    X = small_nonnegative_matrix(cells=4, genes=3)

    kernel._nmf_gpu(
        X,
        {"n_components": 2, "max_iter": 1, "random_state": 0},
        {"device": "cpu", "dtype": "fp32", "allow_tf32": True},
    )

    assert captured[-1][0] is False


def test_tf32_scope_is_noop_off_cuda(kernel):
    """The TF32 context manager should leave backend globals untouched off CUDA."""
    class FakeTorch:
        def __init__(self):
            self.cuda = SimpleNamespace(is_available=lambda: True)
            self.backends = SimpleNamespace(
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))
            )
            self.precision = "highest"

        def get_float32_matmul_precision(self):
            return self.precision

        def set_float32_matmul_precision(self, value):
            self.precision = value

    fake = FakeTorch()

    with kernel._cuda_tf32(fake, True, "cpu"):
        assert fake.backends.cuda.matmul.allow_tf32 is False
        assert fake.precision == "highest"

    assert fake.backends.cuda.matmul.allow_tf32 is False
    assert fake.precision == "highest"


def test_cuda_fp32_smoke_when_gpu_available(kernel):
    """CUDA fp32 should run when PyTorch reports CUDA available; no CUDA version is pinned here."""
    torch = require_nmf_runtime()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    X = small_nonnegative_matrix(cells=6, genes=5)

    H, W = kernel._nmf_gpu(
        X,
        {"n_components": 2, "max_iter": 2, "random_state": 0},
        {"device": "cuda", "dtype": "fp32"},
    )

    assert_valid_nmf_output(X, H, W, 2)


def test_cuda_bf16_uses_bf16_storage_and_matmul_when_gpu_available(kernel, monkeypatch):
    """CUDA bf16 should require CUDA plus PyTorch-reported bf16 device support."""
    torch = require_nmf_runtime()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bf16")
    captured = []

    def fake_fit_mu(torch, Xg, W, H, eps, max_iter, tol, step, block, tf32, device):
        captured.append((Xg.dtype, W.dtype, H.dtype, eps.dtype, tf32, device))
        return W, H

    monkeypatch.setattr(kernel, "_fit_mu", fake_fit_mu)
    X = small_nonnegative_matrix(cells=4, genes=3)

    kernel._nmf_gpu(
        X,
        {"n_components": 2, "max_iter": 1, "random_state": 0},
        {"device": "cuda", "dtype": "bf16"},
    )

    assert captured == [(torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.bfloat16, False, "cuda")]


def test_cuda_allow_tf32_scope_restores_previous_state_when_gpu_available(kernel):
    """CUDA TF32 should require CUDA availability and restore global torch matmul settings."""
    torch = require_nmf_runtime()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    prev_allow = torch.backends.cuda.matmul.allow_tf32
    prev_precision = torch.get_float32_matmul_precision()

    with kernel._cuda_tf32(torch, not prev_allow, "cuda"):
        assert torch.backends.cuda.matmul.allow_tf32 is (not prev_allow)

    assert torch.backends.cuda.matmul.allow_tf32 is prev_allow
    assert torch.get_float32_matmul_precision() == prev_precision


# ---------------------------------------------------------------------
# Batched-replicate factorize (--gpu-batch): batch-aware kernel parity
# ---------------------------------------------------------------------
def _rowwise_cosine(A, B):
    """Cosine of each aligned program row (same seed -> same init -> same row order, no permutation)."""
    num = (A * B).sum(axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-30
    return num / den


def test_nmf_gpu_mu_matches_single_kernel_for_each_seed(kernel):
    """Each batched replicate should match the single-replicate kernel at the same seed."""
    require_nmf_runtime()
    k = 3
    X = low_rank_matrix(rank=k)
    seeds = [7, 3, 101]
    nmf_kwargs = {"n_components": k, "max_iter": 300, "tol": 0}
    gpu_kwargs = {"device": "cpu"}

    batched = kernel._nmf_gpu_mu(X, seeds, nmf_kwargs, gpu_kwargs)
    assert len(batched) == len(seeds)

    for (Hb, Wb), s in zip(batched, seeds):
        Hs, Ws = kernel._nmf_gpu(X, dict(nmf_kwargs, random_state=s), gpu_kwargs)
        assert Hb.shape == Hs.shape and Wb.shape == Ws.shape
        assert _rowwise_cosine(Hb, Hs).min() > 0.9999
        rel_b = np.linalg.norm(X - Wb @ Hb) / np.linalg.norm(X)
        rel_s = np.linalg.norm(X - Ws @ Hs) / np.linalg.norm(X)
        assert abs(rel_b - rel_s) < 1e-4


def test_nmf_gpu_mu_single_seed_reduces_to_single_kernel(kernel):
    """A batch of one (R=1) must reproduce the single-replicate result at that seed."""
    require_nmf_runtime()
    X = small_nonnegative_matrix(cells=12, genes=6)
    kw = {"n_components": 2, "max_iter": 80, "tol": 0}

    (Hb, Wb), = kernel._nmf_gpu_mu(X, [5], kw, {"device": "cpu"})
    Hs, Ws = kernel._nmf_gpu(X, dict(kw, random_state=5), {"device": "cpu"})

    assert Hb.shape == Hs.shape and Wb.shape == Ws.shape
    assert _rowwise_cosine(Hb, Hs).min() > 0.9999


def test_nmf_gpu_mu_distinct_seeds_give_distinct_replicates(kernel):
    """Distinct seeds should keep replicate outputs distinct."""
    require_nmf_runtime()
    X = small_nonnegative_matrix(cells=20, genes=8)

    out = kernel._nmf_gpu_mu(X, [1, 2], {"n_components": 3, "max_iter": 50}, {"device": "cpu"})

    assert not np.allclose(out[0][0], out[1][0])


def test_nmf_gpu_mu_rejects_empty_seeds(kernel):
    """An empty seed list is a caller error (no replicates to run), not a silent no-op."""
    require_nmf_runtime()
    X = small_nonnegative_matrix(cells=6, genes=4)

    with pytest.raises(ValueError, match="non-empty"):
        kernel._nmf_gpu_mu(X, [], {"n_components": 2, "max_iter": 1}, {"device": "cpu"})


def test_nmf_gpu_mu_results_are_invariant_to_batch_grouping(kernel):
    """Changing seed grouping should not change per-seed factors."""
    require_nmf_runtime()
    X = small_nonnegative_matrix(cells=16, genes=7)
    seeds = [4, 11, 29]
    kw = {"n_components": 3, "max_iter": 150, "tol": 0}
    gpu = {"device": "cpu"}

    batched = kernel._nmf_gpu_mu(X, seeds, kw, gpu)                  # one launch, R=3
    assert len(batched) == len(seeds)

    for (Hb, Wb), s in zip(batched, seeds):
        (Hs, Ws), = kernel._nmf_gpu_mu(X, [s], kw, gpu)             # its own launch, R=1
        assert Hb.shape == Hs.shape and Wb.shape == Ws.shape
        assert _rowwise_cosine(Hb, Hs).min() > 0.9999
        assert _rowwise_cosine(Wb.T, Ws.T).min() > 0.9999


def test_fit_mu_is_batch_aware_and_stops_when_all_slices_converge(kernel):
    """Batched `_fit_mu` should stop only after all slices meet tolerance."""
    torch = require_nmf_runtime()
    R = 3
    Xb = torch.full((1, 3, 2), 2.0, dtype=torch.float64)
    W = torch.ones((R, 3, 1), dtype=torch.float64)
    H = torch.ones((R, 1, 2), dtype=torch.float64)
    eps = torch.tensor(1e-9, dtype=torch.float64)
    calls = {"count": 0}

    def no_change_step(W, H, Xg, eps):
        calls["count"] += 1
        return W, H

    Wout, Hout = kernel._fit_mu(torch, Xb, W, H, eps, 10, 1e-4, no_change_step, 1, False, "cpu")

    assert calls["count"] == 2
    assert Wout.shape == (R, 3, 1) and Hout.shape == (R, 1, 2)


# ---------------------------------------------------------------------
# Fixed-H consensus refit (batch-aware): keep H fixed, update W only
# ---------------------------------------------------------------------
def _fixed_spectra(k, genes, seed=0):
    """A valid non-negative fixed H (spectra) of shape (k, genes)."""
    return np.abs(np.random.default_rng(seed).standard_normal((k, genes)))


def test_nmf_gpu_fixed_h_keeps_spectra_fixed_and_updates_usages(kernel):
    """Fixed-H refit should return the supplied H unchanged."""
    require_nmf_runtime()
    X = small_nonnegative_matrix(cells=10, genes=6)
    k, Hfix = 3, _fixed_spectra(3, 6)
    kw = {"n_components": k, "max_iter": 50, "H": Hfix, "update_H": False}

    (H, W), = kernel._nmf_gpu_fixed_h(X, [7], kw, {"device": "cpu"})

    assert H.shape == (k, 6) and W.shape == (10, k)
    assert np.allclose(H, Hfix)


def test_nmf_gpu_fixed_h_batched_matches_single_refit_per_seed(kernel):
    """Each batched fixed-H refit slice should match a single-seed refit."""
    require_nmf_runtime()
    X = small_nonnegative_matrix(cells=12, genes=5)
    k, Hfix, seeds = 2, _fixed_spectra(2, 5, seed=1), [3, 9]
    kw = {"n_components": k, "max_iter": 100, "tol": 0, "H": Hfix, "update_H": False}

    batched = kernel._nmf_gpu_fixed_h(X, seeds, kw, {"device": "cpu"})
    assert len(batched) == len(seeds)

    for (Hb, Wb), s in zip(batched, seeds):
        (Hs, Ws), = kernel._nmf_gpu_fixed_h(X, [s], kw, {"device": "cpu"})
        assert np.allclose(Hb, Hfix) and np.allclose(Hs, Hfix)
        assert _rowwise_cosine(Wb.T, Ws.T).min() > 0.9999


def test_nmf_gpu_fixed_h_distinct_seeds_give_distinct_usages(kernel):
    """Distinct W initializations should keep fixed-H usage outputs distinct."""
    require_nmf_runtime()
    X = small_nonnegative_matrix(cells=14, genes=6)
    k, Hfix = 3, _fixed_spectra(3, 6, seed=2)
    kw = {"n_components": k, "max_iter": 40, "H": Hfix, "update_H": False}

    out = kernel._nmf_gpu_fixed_h(X, [1, 2], kw, {"device": "cpu"})

    assert not np.allclose(out[0][1], out[1][1])


def test_nmf_gpu_update_H_false_dispatches_to_fixed_h_refit(kernel):
    """`_nmf_gpu(update_H=False)` should dispatch to the fixed-H refit."""
    require_nmf_runtime()
    X = small_nonnegative_matrix(cells=8, genes=4)
    k, Hfix = 2, _fixed_spectra(2, 4, seed=3)

    H, W = kernel._nmf_gpu(X, {"n_components": k, "max_iter": 20, "H": Hfix, "update_H": False}, {"device": "cpu"})

    assert np.allclose(H, Hfix) and W.shape == (8, k)


# ---------------------------------------------------------------------
# --gpu-batch config plumbing
# ---------------------------------------------------------------------
def test_default_gpu_batch_is_single_replicate(kernel):
    """The default batch is 1, so the single-replicate path is unchanged unless a user opts in."""
    assert kernel.DEFAULT_GPU["batch"] == 1
    assert kernel._resolve_gpu_opts(None)["batch"] == 1


def test_resolve_gpu_opts_coerces_and_floors_batch_to_at_least_one(kernel):
    """batch is a positive int: numeric strings coerce and non-positive values floor to 1."""
    assert kernel._resolve_gpu_opts({"batch": "4"})["batch"] == 4
    assert kernel._resolve_gpu_opts({"batch": 0})["batch"] == 1
    assert kernel._resolve_gpu_opts({"batch": -5})["batch"] == 1


def test_parse_gpu_args_registers_batch_and_gpu_kwargs_carries_it(kernel):
    """--gpu-batch parses under --engine gpu and flows into resolved gpu_kwargs; absent -> default 1."""
    import argparse
    parser = kernel.parse_gpu_args(argparse.ArgumentParser())

    args = parser.parse_args(["--engine", "gpu", "--gpu-batch", "8"])
    assert args.gpu_batch == 8
    assert kernel.gpu_kwargs_from_args(args)["batch"] == 8

    default_args = parser.parse_args(["--engine", "gpu"])
    assert default_args.gpu_batch is None
    assert kernel.gpu_kwargs_from_args(default_args)["batch"] == 1


# ---------------------------------------------------------------------
# CLI parsing, engine wiring, and fixed-H consensus refit (integration)
# ---------------------------------------------------------------------
def test_parse_gpu_args_defaults_to_none_until_user_sets_engine(kernel):
    """CLI flags should default to None so absent options are distinguishable from explicit values."""
    parser = argparse.ArgumentParser()
    kernel.parse_gpu_args(parser)

    args = parser.parse_args([])

    assert args.engine is None
    for name in ("gpu_device", "gpu_dtype", "gpu_allow_tf32", "gpu_compile",
                 "gpu_eps", "gpu_check_every", "gpu_compile_block"):
        assert getattr(args, name) is None, f"{name} should default to None"


def test_gpu_kwargs_from_args_rejects_gpu_options_without_gpu_engine(kernel):
    """GPU-specific CLI options should raise unless `--engine gpu` was explicitly selected."""
    parser = argparse.ArgumentParser()
    kernel.parse_gpu_args(parser)

    for argv in (["--gpu-device", "cuda"], ["--engine", "cpu", "--gpu-device", "cuda"]):
        with pytest.raises(ValueError, match="require --engine gpu"):
            kernel.gpu_kwargs_from_args(parser.parse_args(argv))


def test_gpu_kwargs_from_args_fills_defaults_when_gpu_engine_selected(kernel):
    """`--engine gpu` alone should resolve missing GPU options from DEFAULT_GPU."""
    parser = argparse.ArgumentParser()
    kernel.parse_gpu_args(parser)

    args = parser.parse_args(["--engine", "gpu"])

    assert kernel.gpu_kwargs_from_args(args) == kernel.DEFAULT_GPU


def test_gpu_kwargs_from_args_normalizes_cli_overrides(kernel):
    """GPU CLI override values should normalize through the same resolver as config dict values."""
    parser = argparse.ArgumentParser()
    kernel.parse_gpu_args(parser)

    args = parser.parse_args([
        "--engine", "gpu",
        "--gpu-device", "CUDA:0",        # device is lower-cased by _resolve_gpu_opts
        "--gpu-dtype", "FP32",
        "--gpu-allow-tf32",              # store_const flags -> True
        "--gpu-compile",
        "--gpu-eps", "1e-8",
        "--gpu-check-every", "5",
        "--gpu-compile-block", "100",
    ])

    assert kernel.gpu_kwargs_from_args(args) == {
        "device": "cuda:0",
        "dtype": "fp32",
        "allow_tf32": True,
        "compile": True,
        "eps": 1e-8,
        "check_every": 5,
        "compile_block": 100,
        "batch": 1,          # default added; not set on the CLI here
    }


def test_validate_engine_args_for_command_rejects_non_engine_command_gpu_options(kernel):
    """Engine/GPU options should be accepted only for factorize and consensus."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    kernel.parse_gpu_args(parser)
    supported = ("factorize", "consensus")

    # factorize and consensus accept engine/GPU options (no raise)
    for argv in (
        ["factorize", "--engine", "gpu"],
        ["consensus", "--engine", "gpu"],
        ["consensus", "--gpu-device", "cuda"],
    ):
        kernel.validate_engine_args_for_command(parser.parse_args(argv), supported)

    # non-engine commands carrying engine/GPU options are rejected
    for argv in (
        ["prepare", "--engine", "gpu"],
        ["combine", "--gpu-device", "cuda"],
        ["k_selection_plot", "--gpu-dtype", "fp32"],
    ):
        with pytest.raises(ValueError, match="only valid with"):
            kernel.validate_engine_args_for_command(parser.parse_args(argv), supported)

    # non-engine commands without engine/GPU options are fine
    kernel.validate_engine_args_for_command(parser.parse_args(["prepare"]), supported)


def test_validate_engine_args_accepts_consensus_gpu_options(kernel):
    """`consensus --engine gpu` and consensus GPU flags should be valid CLI input."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    kernel.parse_gpu_args(parser)
    supported = ("factorize", "consensus")

    for argv in (
        ["consensus", "--engine", "gpu"],
        ["consensus", "--engine", "gpu", "--gpu-device", "CUDA:0", "--gpu-dtype", "FP32"],
        ["consensus", "--gpu-allow-tf32", "--gpu-compile"],
    ):
        kernel.validate_engine_args_for_command(parser.parse_args(argv), supported)


def test_validate_engine_args_rejects_gpu_options_for_non_engine_commands(kernel):
    """GPU flags should still be rejected for prepare/combine/k_selection_plot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    kernel.parse_gpu_args(parser)
    supported = ("factorize", "consensus")

    for argv in (
        ["prepare", "--engine", "gpu"],
        ["combine", "--gpu-device", "cuda"],
        ["k_selection_plot", "--gpu-check-every", "2"],
    ):
        with pytest.raises(ValueError, match="only valid with"):
            kernel.validate_engine_args_for_command(parser.parse_args(argv), supported)


def test_configure_nmf_engine_cpu_leaves_cnmf_instance_unchanged(kernel):
    """The default CPU engine should be a no-op so existing sklearn behavior is preserved."""
    class DummyCNMF:
        def _nmf(self, X, nmf_kwargs):
            return "sklearn-path"

    obj = DummyCNMF()
    result = kernel.configure_nmf_engine(obj, engine="cpu", gpu_kwargs={"device": "cuda"})

    assert result is obj
    assert "_nmf" not in vars(obj)                  # no instance override added
    assert obj._nmf("X", {}) == "sklearn-path"


def test_configure_nmf_engine_rejects_unknown_engine(kernel):
    """Unknown engine names should fail loudly instead of silently using CPU."""
    with pytest.raises(ValueError, match="engine must be 'cpu' or 'gpu'"):
        kernel.configure_nmf_engine(object(), engine="tpu")


def test_configure_nmf_engine_gpu_patches_instance_nmf_with_adapter(kernel, monkeypatch):
    """The GPU engine should replace the instance `_nmf` callable with the GPU adapter path."""
    captured = {}

    def fake_nmf_gpu(self, X, nmf_kwargs):
        captured["self"], captured["X"], captured["nmf_kwargs"] = self, X, dict(nmf_kwargs)
        return ("spectra", "usages")

    monkeypatch.setattr(kernel, "nmf_gpu", fake_nmf_gpu)

    class DummyCNMF:
        def _nmf(self, X, nmf_kwargs):
            return "sklearn-path"

    obj = DummyCNMF()
    gpu_kwargs = {"device": "cuda", "dtype": "fp32"}
    result = kernel.configure_nmf_engine(obj, engine="gpu", gpu_kwargs=gpu_kwargs)

    assert result is obj
    assert "_nmf" in vars(obj)                                  # instance _nmf is now overridden

    out = obj._nmf("Xdata", {"n_components": 5})

    assert out == ("spectra", "usages")                        # dispatched through the GPU adapter
    assert captured["self"] is obj and captured["X"] == "Xdata"
    assert captured["nmf_kwargs"]["engine"] == "gpu"           # engine + gpu kwargs embedded first
    assert captured["nmf_kwargs"]["gpu"] == gpu_kwargs
    assert captured["nmf_kwargs"]["n_components"] == 5


def test_nmf_gpu_update_h_false_reconstructs_and_keeps_fixed_h(kernel):
    """Consensus refit should preserve supplied H and optimize usages W only."""
    require_nmf_runtime()
    fixed_h = np.array([[1.0, 0.3, 0.6], [0.2, 1.1, 0.4]], dtype=np.float64)
    true_w = np.array([[1.0, 0.5], [0.4, 1.2], [1.5, 0.3], [0.7, 0.9]], dtype=np.float64)
    X = true_w @ fixed_h

    H, W = kernel._nmf_gpu(
        X,
        {"n_components": 2, "max_iter": 5, "random_state": 0, "update_H": False, "H": fixed_h},
        {"device": "cpu", "dtype": "fp64", "check_every": 5},
    )

    assert_valid_nmf_output(X, H, W, 2)
    assert np.allclose(H, fixed_h)


def test_to_checked_fixed_h_rejects_missing_invalid_or_incompatible_h(kernel):
    """Fixed-H consensus refit should fail clearly for invalid supplied spectra."""
    with pytest.raises(ValueError, match="requires a fixed H"):
        kernel._to_checked_fixed_h(None, 2, 3)

    invalid_cases = [
        (np.array([1.0, 2.0, 3.0]), "2D"),
        (np.ones((3, 3)), "shape"),
        (np.array([[1.0, np.nan, 0.2], [0.4, 0.5, 0.6]]), "NaN/inf"),
        (np.array([[1.0, -0.1, 0.2], [0.4, 0.5, 0.6]]), "non-negative"),
    ]
    for H, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            kernel._to_checked_fixed_h(H, 2, 3)


def test_mu_step_fixed_h_matches_manual_w_only_update(kernel):
    """One fixed-H MU step should match the manual W-only Frobenius update."""
    torch = require_nmf_runtime()
    Xg = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    W0 = torch.tensor([[0.5, 0.7], [0.9, 1.1]], dtype=torch.float64)
    H = torch.tensor([[0.6, 0.8], [1.0, 1.2]], dtype=torch.float64)
    eps = torch.tensor(1e-9, dtype=torch.float64)
    H_before = H.clone()

    denominator = W0 @ (H @ H.T)
    denominator = denominator.where(denominator != 0, eps)
    expected_W = W0 * ((Xg @ H.T) / denominator)
    W = kernel._mu_step_fixed_h(W0, H, Xg, eps)

    assert torch.allclose(W, expected_W)
    assert torch.allclose(H, H_before)


def test_fit_mu_fixed_h_respects_early_stop_and_max_iter_block_bounds(kernel):
    """Fixed-H fit loop should share early-stop and no-overrun guarantees with full MU."""
    torch = require_nmf_runtime()
    Xg = torch.full((3, 2), 2.0, dtype=torch.float64)
    W = torch.ones((3, 1), dtype=torch.float64)
    H = torch.ones((1, 2), dtype=torch.float64)
    eps = torch.tensor(1e-9, dtype=torch.float64)

    early_calls = {"count": 0}

    def no_change_step_early(W, H, Xg, eps):
        early_calls["count"] += 1
        return W

    kernel._fit_mu_fixed_h(torch, Xg, W, H, eps, 10, 1e-4, no_change_step_early, 1, False, "cpu")
    assert early_calls["count"] == 2

    max_iter_calls = {"count": 0}

    def no_change_step_max_iter(W, H, Xg, eps):
        max_iter_calls["count"] += 1
        return W

    kernel._fit_mu_fixed_h(torch, Xg, W, H, eps, 6, -1.0, no_change_step_max_iter, 4, False, "cpu")
    assert max_iter_calls["count"] == 6


def test_execution_plan_for_fixed_h_compile_uses_fixed_h_step_and_compile_block(kernel):
    """Compiled consensus refit should compile _mu_step_fixed_h and use compile_block."""
    calls = []
    fake_torch = SimpleNamespace(compile=lambda fn: calls.append(fn) or fn)
    opt = dict(kernel.DEFAULT_GPU, compile=True, check_every=1, compile_block=3)

    step, block = kernel._execution_plan(fake_torch, opt, "cpu", kernel._mu_step_fixed_h)

    assert calls == [kernel._mu_step_fixed_h]
    assert step is kernel._mu_step_fixed_h
    assert block == 3

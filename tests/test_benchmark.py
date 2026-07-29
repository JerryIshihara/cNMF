"""Controlled backend benchmark tests with explicit shared initialization."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cnmf.benchmark import (
    run_cpu_cd,
    run_cpu_mu_checkpoints,
    run_gpu_cd_checkpoints,
    run_gpu_mu_checkpoints,
    run_torchnmf_checkpoints,
)


CASES = [
    pytest.param("fp32", np.float32, 3e-5, 3e-6, id="fp32"),
    pytest.param("fp64", np.float64, 3e-10, 3e-11, id="fp64"),
]


def initialized_case(np_dtype, seed=17):
    from sklearn.decomposition._nmf import _initialize_nmf

    rng = np.random.default_rng(314)
    X = rng.random((37, 29), dtype=np_dtype) + np_dtype(0.1)
    W0, H0 = _initialize_nmf(
        X,
        n_components=5,
        init="random",
        random_state=seed,
    )
    return X, W0, H0


@pytest.mark.parametrize("dtype_name,np_dtype,rtol,atol", CASES)
def test_cpu_mu_checkpoints_match_sklearn_exactly(
    dtype_name, np_dtype, rtol, atol
):
    from sklearn.decomposition import non_negative_factorization
    from sklearn.exceptions import ConvergenceWarning

    X, W0, H0 = initialized_case(np_dtype)
    snapshots = {}
    actual_W, actual_H = run_cpu_mu_checkpoints(
        X,
        W0,
        H0,
        checkpoints=(1, 3, 5),
        callback=lambda iteration, W, H: snapshots.update(
            {iteration: (W.copy(), H.copy())}
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        expected_W, expected_H, iterations = non_negative_factorization(
            X,
            W=W0.copy(),
            H=H0.copy(),
            n_components=H0.shape[0],
            init="custom",
            solver="mu",
            beta_loss="frobenius",
            tol=0.0,
            max_iter=5,
            alpha_W=0.0,
            alpha_H=0.0,
            l1_ratio=0.0,
            random_state=17,
        )
    assert iterations == 5
    assert sorted(snapshots) == [1, 3, 5]
    np.testing.assert_array_equal(actual_W, expected_W)
    np.testing.assert_array_equal(actual_H, expected_H)


@pytest.mark.parametrize("dtype_name,np_dtype,rtol,atol", CASES)
def test_cpu_cd_uses_the_supplied_factors(dtype_name, np_dtype, rtol, atol):
    X, W0, H0 = initialized_case(np_dtype)
    result_a = run_cpu_cd(X, W0, H0, seed=17, max_iter=5)
    result_b = run_cpu_cd(X, W0, H0, seed=999, max_iter=5)
    assert result_a[2] == result_b[2] == 5
    np.testing.assert_array_equal(result_a[0], result_b[0])
    np.testing.assert_array_equal(result_a[1], result_b[1])


@pytest.mark.parametrize("dtype_name,np_dtype,rtol,atol", CASES)
def test_gpu_cd_checkpoint_runner_matches_cpu_cd_from_persisted_factors(
    dtype_name, np_dtype, rtol, atol
):
    from sklearn.decomposition._nmf import _initialize_nmf

    X, W0_a, H0_a = initialized_case(np_dtype, seed=17)
    W0_b, H0_b = _initialize_nmf(
        X,
        n_components=H0_a.shape[0],
        init="random",
        random_state=29,
    )
    W0 = np.stack((W0_a, W0_b))
    H0 = np.stack((H0_a, H0_b))
    snapshots = {}

    def capture(iteration, Xg, Wg, Hg, state):
        assert Xg.device.type == Wg.device.type == Hg.device.type == "cpu"
        assert state["iteration_seconds"]
        snapshots[iteration] = (
            Wg.detach().cpu().numpy().copy(),
            Hg.detach().cpu().numpy().copy(),
        )

    metrics = run_gpu_cd_checkpoints(
        X,
        W0,
        H0,
        checkpoints=(1, 3, 5),
        callback=capture,
        device="cpu",
        dtype=dtype_name,
        seeds=(17, 29),
    )

    assert sorted(snapshots) == [1, 3, 5]
    assert metrics["completed_iterations"] == 5
    assert metrics["checkpoints_reached"] == [1, 3, 5]
    assert len(metrics["iteration_seconds"]) == 5
    assert len(metrics["active_replicates_per_iteration"]) == 5
    assert all(value > 0 for value in metrics["iteration_seconds"])
    assert metrics["timing"] == "raw_per_iteration_perf_counter"
    assert metrics["timing_unit"] == "seconds"
    assert metrics["timing_is_amortized"] is False
    assert metrics["tf32"] is False
    assert metrics["tol"] == 0.0

    for iteration, (actual_W, actual_H) in snapshots.items():
        for replicate, seed in enumerate((17, 29)):
            expected_W, expected_H, expected_iterations = run_cpu_cd(
                X,
                W0[replicate],
                H0[replicate],
                seed=seed,
                max_iter=iteration,
            )
            assert expected_iterations == iteration
            np.testing.assert_allclose(
                actual_W[replicate], expected_W, rtol=rtol, atol=atol
            )
            np.testing.assert_allclose(
                actual_H[replicate], expected_H, rtol=rtol, atol=atol
            )


def test_gpu_cd_warmup_does_not_change_the_persisted_start():
    X, W0, H0 = initialized_case(np.float64)
    outputs = {}

    for warmup in (False, True):
        run_gpu_cd_checkpoints(
            X,
            W0[None, ...],
            H0[None, ...],
            checkpoints=(4,),
            callback=lambda iteration, Xg, Wg, Hg, state, warmup=warmup: (
                outputs.update(
                    {
                        warmup: (
                            Wg.detach().cpu().numpy().copy(),
                            Hg.detach().cpu().numpy().copy(),
                        )
                    }
                )
            ),
            device="cpu",
            dtype="fp64",
            seeds=(17,),
            warmup=warmup,
        )

    np.testing.assert_array_equal(outputs[False][0], outputs[True][0])
    np.testing.assert_array_equal(outputs[False][1], outputs[True][1])


def test_gpu_cd_checkpoint_runner_records_early_inactive_replicates():
    X = np.zeros((6, 4), dtype=np.float32)
    W0 = np.zeros((2, 6, 2), dtype=np.float32)
    H0 = np.zeros((2, 2, 4), dtype=np.float32)
    seen = []
    metrics = run_gpu_cd_checkpoints(
        X,
        W0,
        H0,
        checkpoints=(1, 3),
        callback=lambda iteration, Xg, Wg, Hg, state: seen.append(
            (iteration, state["active_replicates"])
        ),
        device="cpu",
        dtype="fp32",
        seeds=(1, 2),
    )

    assert seen == [(1, 0)]
    assert metrics["completed_iterations"] == 1
    assert metrics["checkpoints_reached"] == [1]
    assert metrics["active_replicates_per_iteration"] == [0]
    assert metrics["n_iter"] == [1, 1]


def test_gpu_cd_checkpoint_runner_uses_raw_cuda_event_timings():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    X, W0, H0 = initialized_case(np.float32)
    snapshots = {}
    metrics = run_gpu_cd_checkpoints(
        X,
        np.stack((W0, W0)),
        np.stack((H0, H0)),
        checkpoints=(1, 3),
        callback=lambda iteration, Xg, Wg, Hg, state: snapshots.update(
            {
                iteration: (
                    Wg.detach().cpu().numpy().copy(),
                    Hg.detach().cpu().numpy().copy(),
                )
            }
        ),
        device="cuda",
        dtype="fp32",
        seeds=(17, 17),
    )

    assert sorted(snapshots) == [1, 3]
    assert metrics["timing"] == "raw_per_iteration_cuda_event"
    assert metrics["timing_unit"] == "seconds"
    assert metrics["timing_is_amortized"] is False
    assert len(metrics["iteration_seconds"]) == 3
    assert len(metrics["active_replicates_per_iteration"]) == 3
    assert all(value > 0 for value in metrics["iteration_seconds"])
    np.testing.assert_array_equal(snapshots[3][0][0], snapshots[3][0][1])
    np.testing.assert_array_equal(snapshots[3][1][0], snapshots[3][1][1])


@pytest.mark.parametrize("dtype_name,np_dtype,rtol,atol", CASES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_gpu_mu_matches_cpu_mu_from_identical_factors(
    dtype_name, np_dtype, rtol, atol, device
):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    X, W0, H0 = initialized_case(np_dtype)
    cpu_snapshots = {}
    run_cpu_mu_checkpoints(
        X,
        W0,
        H0,
        checkpoints=(1, 3, 5),
        callback=lambda iteration, W, H: cpu_snapshots.update(
            {iteration: (W.copy(), H.copy())}
        ),
    )
    gpu_snapshots = {}
    run_gpu_mu_checkpoints(
        X,
        np.stack((W0, W0)),
        np.stack((H0, H0)),
        checkpoints=(1, 3, 5),
        callback=lambda iteration, W, H: gpu_snapshots.update(
            {iteration: (W.copy(), H.copy())}
        ),
        device=device,
        dtype=dtype_name,
        allow_tf32=False,
    )

    assert sorted(gpu_snapshots) == [1, 3, 5]
    for iteration in gpu_snapshots:
        expected_W, expected_H = cpu_snapshots[iteration]
        actual_W, actual_H = gpu_snapshots[iteration]
        np.testing.assert_array_equal(actual_W[0], actual_W[1])
        np.testing.assert_array_equal(actual_H[0], actual_H[1])
        if device == "cpu":
            np.testing.assert_array_equal(actual_W[0], expected_W)
            np.testing.assert_array_equal(actual_H[0], expected_H)
        else:
            np.testing.assert_allclose(
                actual_W[0], expected_W, rtol=rtol, atol=atol
            )
            np.testing.assert_allclose(
                actual_H[0], expected_H, rtol=rtol, atol=atol
            )


@pytest.mark.parametrize("dtype_name,np_dtype,rtol,atol", CASES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torchnmf_matches_cpu_mu_order_and_checkpoints(
    dtype_name, np_dtype, rtol, atol, device
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchnmf")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    X, W0, H0 = initialized_case(np_dtype)
    cpu_snapshots = {}
    run_cpu_mu_checkpoints(
        X,
        W0,
        H0,
        checkpoints=(1, 3, 5),
        callback=lambda iteration, W, H: cpu_snapshots.update(
            {iteration: (W.copy(), H.copy())}
        ),
    )
    torch_snapshots = {}
    run_torchnmf_checkpoints(
        X,
        W0,
        H0,
        checkpoints=(1, 3, 5),
        callback=lambda iteration, W, H: torch_snapshots.update(
            {iteration: (W.copy(), H.copy())}
        ),
        device=device,
        dtype=dtype_name,
    )

    assert sorted(torch_snapshots) == [1, 3, 5]
    for iteration in torch_snapshots:
        expected_W, expected_H = cpu_snapshots[iteration]
        actual_W, actual_H = torch_snapshots[iteration]
        np.testing.assert_allclose(actual_W, expected_W, rtol=rtol, atol=atol)
        np.testing.assert_allclose(actual_H, expected_H, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype_name,np_dtype,rtol,atol", CASES)
def test_gpu_and_torchnmf_zero_denominator_guard_is_finite(
    dtype_name, np_dtype, rtol, atol
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchnmf")

    X = np.zeros((6, 4), dtype=np_dtype)
    W0 = np.zeros((6, 2), dtype=np_dtype)
    H0 = np.zeros((2, 4), dtype=np_dtype)
    outputs = []
    run_gpu_mu_checkpoints(
        X,
        W0[None, ...],
        H0[None, ...],
        checkpoints=(1,),
        callback=lambda iteration, W, H: outputs.extend((W, H)),
        device="cpu",
        dtype=dtype_name,
    )
    run_torchnmf_checkpoints(
        X,
        W0,
        H0,
        checkpoints=(1,),
        callback=lambda iteration, W, H: outputs.extend((W, H)),
        device="cpu",
        dtype=dtype_name,
    )
    assert outputs
    assert all(np.isfinite(value).all() for value in outputs)
    assert all(np.count_nonzero(value) == 0 for value in outputs)

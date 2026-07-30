import pytest
import numpy as np
import pandas as pd
import scanpy as sc
import os
import scipy.sparse as sp
import yaml
from cnmf import cNMF, save_df_to_npz, load_df_from_npz
from cnmf import nmf_gpu

# Global parameters for data simulation
NUM_CELLS = 100
NUM_GENES = 500
BINOM_N = 100
BINOM_P = 0.01
SEED = 42

@pytest.fixture
def mock_cnmf(tmp_path):
    return cNMF(output_dir=str(tmp_path), name="test")

def generate_counts_file(tmp_path, file_format, dtype=np.int64, zero_count=False):
    """
    Generates a synthetic single-cell RNA-seq counts file in various formats.
    
    Args:
        tmp_path (Path): Temporary path for storing the file.
        file_format (str): One of ['txt', 'npz', 'h5ad'].
        dtype (numpy dtype, optional): The data type to store (int64, float32, etc.).
        zero_count (bool, optional): If True, makes the first cell have zero counts.

    Returns:
        str: Path to the generated counts file.
    """
    np.random.seed(SEED)
    data = np.random.binomial(n=BINOM_N, p=BINOM_P, size=(NUM_CELLS, NUM_GENES)).astype(dtype)
    
    if zero_count:
        data[0, :] = 0  # Introduce zero-count cells

    if file_format == "txt":
        df = pd.DataFrame(data, columns=[f"gene{i}" for i in range(NUM_GENES)],
                          index=[f"cell{i}" for i in range(NUM_CELLS)])
        counts_fn = tmp_path / f"counts_{dtype.__name__}.txt"
        df.to_csv(counts_fn, sep='\t')

    elif file_format == "npz":
        df = pd.DataFrame(data, columns=[f"gene{i}" for i in range(NUM_GENES)],
                          index=[f"cell{i}" for i in range(NUM_CELLS)])
        counts_fn = tmp_path / f"counts_{dtype.__name__}.npz"
        save_df_to_npz(df, counts_fn)

    elif file_format == "h5ad":
        adata = sc.AnnData(X=sp.csr_matrix(data))
        counts_fn = tmp_path / f"counts_{dtype.__name__}.h5ad"
        adata.write_h5ad(counts_fn)

    else:
        raise ValueError("Unsupported file format. Choose from ['txt', 'npz', 'h5ad'].")

    return str(counts_fn)

@pytest.mark.parametrize("file_format", ["txt", "npz", "h5ad"])
@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize("densify", [True, False])
def test_prepare(mock_cnmf, file_format, dtype, densify, tmp_path):
    counts_fn = generate_counts_file(tmp_path, file_format, dtype)
    
    output_dir = tmp_path / "output"
    os.makedirs(output_dir, exist_ok=True)
    
    mock_cnmf.prepare(counts_fn, components=[5, 10], n_iter=10, densify=densify)
    
    # Check if output files were created
    expected_files = [
        mock_cnmf.paths['normalized_counts'],
        mock_cnmf.paths['nmf_replicate_parameters'],
        mock_cnmf.paths['nmf_run_parameters'],
        mock_cnmf.paths['nmf_genes_list'],
        mock_cnmf.paths['tpm'],
        mock_cnmf.paths['tpm_stats']
    ]
    
    for file in expected_files:
        assert os.path.exists(file), f"Expected output file {file} not found."
    
    # Clean up after test
    for file in expected_files:
        os.remove(file)

@pytest.mark.parametrize("file_format", ["txt", "npz", "h5ad"])
@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize("densify", [True, False])
def test_prepare_raises_on_zero_count_cells(mock_cnmf, file_format, dtype, densify, tmp_path):
    counts_fn = generate_counts_file(tmp_path, file_format, dtype, zero_count=True)

    with pytest.raises(Exception, match="Error: .* cells have zero counts of overdispersed genes.*"):
        mock_cnmf.prepare(counts_fn, components=[5, 10], n_iter=10, densify=densify)


def test_configure_nmf_engine_gpu_factorize_groups_by_k_and_writes_each_replicate(mock_cnmf, monkeypatch, tmp_path):
    """GPU factorize groups by k, batches seeds, and writes one spectra file per replicate."""
    counts_fn = generate_counts_file(tmp_path, "npz", np.float64)
    mock_cnmf.prepare(counts_fn, components=[5, 7], n_iter=3, densify=True)

    calls = []

    def fake_mu(X, seeds, nmf_kwargs, gpu_kwargs=None):
        k = int(nmf_kwargs["n_components"])
        calls.append((k, [int(s) for s in seeds]))
        return [(np.zeros((k, X.shape[1])), np.zeros((X.shape[0], k))) for _ in seeds]

    monkeypatch.setattr(nmf_gpu, "_nmf_gpu_mu", fake_mu)

    nmf_gpu.configure_nmf_engine(mock_cnmf, engine="gpu", gpu_kwargs={"device": "cpu", "batch": 2})
    mock_cnmf.factorize(worker_i=0, total_workers=1)

    # Two k-values x three replicates, chunked by batch=2 -> four launches.
    assert len(calls) == 4
    assert sorted(len(seeds) for _, seeds in calls) == [1, 1, 2, 2]
    for k, seeds in calls:
        assert k in (5, 7)
        assert 1 <= len(seeds) <= 2

    run_params = load_df_from_npz(mock_cnmf.paths["nmf_replicate_parameters"])
    for _, row in run_params.iterrows():
        path = mock_cnmf.paths["iter_spectra"] % (int(row["n_components"]), int(row["iter"]))
        assert os.path.exists(path), f"missing iter_spectra for k={row['n_components']} iter={row['iter']}"

    handed = sorted(s for _, seeds in calls for s in seeds)
    assert handed == sorted(int(s) for s in run_params["nmf_seed"])


def test_configure_nmf_engine_installs_gpu_factorize_at_default_batch_1(mock_cnmf, monkeypatch, tmp_path):
    """Default GPU factorize uses batch=1: one seed per `_nmf_gpu_mu` launch."""
    counts_fn = generate_counts_file(tmp_path, "npz", np.float64)
    mock_cnmf.prepare(counts_fn, components=[6], n_iter=2, densify=True)

    calls = []

    def fake_mu(X, seeds, nmf_kwargs, gpu_kwargs=None):
        calls.append([int(s) for s in seeds])
        k = int(nmf_kwargs["n_components"])
        return [(np.zeros((k, X.shape[1])), np.zeros((X.shape[0], k))) for _ in seeds]

    monkeypatch.setattr(nmf_gpu, "_nmf_gpu_mu", fake_mu)

    nmf_gpu.configure_nmf_engine(mock_cnmf, engine="gpu", gpu_kwargs={"device": "cpu"})
    mock_cnmf.factorize(worker_i=0, total_workers=1)

    assert len(calls) == 2
    assert all(len(seeds) == 1 for seeds in calls)
    run_params = load_df_from_npz(mock_cnmf.paths["nmf_replicate_parameters"])
    for _, row in run_params.iterrows():
        path = mock_cnmf.paths["iter_spectra"] % (int(row["n_components"]), int(row["iter"]))
        assert os.path.exists(path), f"missing iter_spectra for k={row['n_components']} iter={row['iter']}"


# ---------------------------------------------------------------------
# cNMF <-> GPU engine integration: factorize / refit / consensus routing
# ---------------------------------------------------------------------
def generate_positive_counts_file(tmp_path, cells=24, genes=12, dtype=np.float64):
    """Generate a small dense positive count table for consensus smoke tests."""
    rng = np.random.default_rng(SEED)
    data = rng.poisson(lam=4.0, size=(cells, genes)).astype(dtype) + 1
    df = pd.DataFrame(
        data,
        columns=[f"gene{i}" for i in range(genes)],
        index=[f"cell{i}" for i in range(cells)],
    )
    counts_fn = tmp_path / "positive_counts.txt"
    df.to_csv(counts_fn, sep="\t")
    return str(counts_fn)


def write_minimal_nmf_run_params(cnmf_obj):
    """Write the NMF kwargs file required by refit_usage/refit_spectra."""
    run_params = {
        "alpha_W": 0.0,
        "alpha_H": 0.0,
        "l1_ratio": 0.0,
        "beta_loss": "frobenius",
        "solver": "mu",
        "tol": 1e-4,
        "max_iter": 5,
        "init": "random",
    }
    cnmf_obj.save_nmf_iter_params(pd.DataFrame(), run_params)
    return run_params


def fake_gpu_nmf_output(X, nmf_kwargs):
    """Return deterministic positive factors matching the cNMF `(spectra, usages)` contract."""
    k = int(nmf_kwargs["n_components"])
    fixed_h = np.asarray(nmf_kwargs["H"], dtype=np.float64)
    row_scale = np.linspace(1.0, 2.0, X.shape[0], dtype=np.float64).reshape(-1, 1)
    col_scale = np.arange(1, k + 1, dtype=np.float64).reshape(1, -1)
    usages = row_scale * col_scale
    return fixed_h.copy(), usages


def test_get_nmf_iter_params_default_cpu_engine_does_not_change_sklearn_kwargs(mock_cnmf):
    """Default CPU runs should not add engine/gpu keys to sklearn NMF kwargs."""
    _replicate_params, run_params = mock_cnmf.get_nmf_iter_params(ks=[5, 7], n_iter=3, random_state_seed=14)

    # GPU wiring must not leak into the sklearn factorization kwargs.
    assert "engine" not in run_params
    assert "gpu" not in run_params
    # Only the existing cNMF/sklearn factorization keys are present.
    assert set(run_params) == {"alpha_W", "alpha_H", "l1_ratio", "beta_loss", "solver", "tol", "max_iter", "init"}


def test_get_nmf_iter_params_defaults_to_mu_even_for_frobenius(mock_cnmf):
    """Frobenius loss must no longer silently replace the public default MU solver with CD."""
    _replicate_params, run_params = mock_cnmf.get_nmf_iter_params(
        ks=[5],
        n_iter=1,
        random_state_seed=14,
        beta_loss="frobenius",
    )

    assert run_params["beta_loss"] == "frobenius"
    assert run_params["solver"] == "mu"


def test_prepare_persists_explicit_cd_solver(mock_cnmf, tmp_path):
    """The solver selected at prepare time is cached for factorize/resume."""
    counts_fn = generate_counts_file(tmp_path, "npz", np.float64)

    mock_cnmf.prepare(
        counts_fn,
        components=[5],
        n_iter=1,
        densify=True,
        beta_loss="frobenius",
        solver="cd",
    )

    with open(mock_cnmf.paths["nmf_run_parameters"]) as stream:
        run_params = yaml.safe_load(stream)
    assert run_params["solver"] == "cd"
    assert run_params["beta_loss"] == "frobenius"


def test_cd_rejects_non_frobenius_loss(mock_cnmf):
    """Match sklearn: coordinate descent is defined only for Frobenius loss."""
    with pytest.raises(ValueError, match="frobenius"):
        mock_cnmf.get_nmf_iter_params(
            ks=[5],
            n_iter=1,
            beta_loss="kullback-leibler",
            solver="cd",
        )


@pytest.mark.parametrize("solver, expected_backend", [("mu", "mu"), ("cd", "cd")])
def test_factorize_gpu_dispatches_saved_solver(
    mock_cnmf, monkeypatch, tmp_path, capsys, solver, expected_backend
):
    """GPU factorize dispatches every same-k batch from the solver cached by prepare."""
    counts_fn = generate_counts_file(tmp_path, "npz", np.float64)
    mock_cnmf.prepare(
        counts_fn,
        components=[5],
        n_iter=2,
        densify=True,
        beta_loss="frobenius",
        solver=solver,
    )

    calls = []

    def fake_backend(name):
        def run(X, seeds, nmf_kwargs, gpu_kwargs=None, **_backend_kwargs):
            calls.append((name, [int(seed) for seed in seeds], dict(nmf_kwargs)))
            k = int(nmf_kwargs["n_components"])
            results = [
                (np.zeros((k, X.shape[1])), np.zeros((X.shape[0], k)))
                for _ in seeds
            ]
            if _backend_kwargs.get("return_metrics"):
                return results, [
                    {
                        "seed": int(seed),
                        "n_iter": 7,
                        "converged": True,
                        "fit_seconds": 0.25,
                    }
                    for seed in seeds
                ]
            return results

        return run

    monkeypatch.setattr(nmf_gpu, "_nmf_gpu_mu", fake_backend("mu"))
    monkeypatch.setattr(nmf_gpu, "_nmf_gpu_cd", fake_backend("cd"))
    nmf_gpu.configure_nmf_engine(
        mock_cnmf,
        engine="gpu",
        gpu_kwargs={"device": "cpu", "batch": 2},
    )

    mock_cnmf.factorize(worker_i=0, total_workers=1)

    assert len(calls) == 1
    backend, seeds, run_params = calls[0]
    assert backend == expected_backend
    assert len(seeds) == 2
    assert run_params["solver"] == solver
    assert run_params["n_components"] == 5
    if solver == "cd":
        for iter_i in range(2):
            metrics_path = (
                mock_cnmf.paths["iter_spectra"] % (5, iter_i)
            ).replace(".spectra.", ".factorize_metrics.")
            metrics = load_df_from_npz(metrics_path)
            assert metrics.loc[0, "n_iter"] == 7
            assert metrics.loc[0, "converged"] == 1
            assert metrics.loc[0, "allow_tf32"] == 0
        output = capsys.readouterr().out
        assert output.count("HALS_REPLICATE") == 2
        assert "status=converged" in output


def test_factorize_gpu_engine_passes_seed_components_run_params_and_gpu_kwargs(mock_cnmf, monkeypatch, tmp_path):
    """factorize_gpu should hand each replicate's seed, n_components, forwarded run params, and the
    resolved GPU kwargs to the batch kernel _nmf_gpu_mu (default batch=1 -> one seed per launch)."""
    import cnmf.nmf_gpu as gpu_mod

    counts_fn = generate_counts_file(tmp_path, "txt", np.int64)
    mock_cnmf.prepare(counts_fn, components=[5], n_iter=2, densify=True, seed=14)

    captured = []

    def fake_mu(X, seeds, nmf_kwargs, gpu_kwargs=None):
        captured.append((list(seeds), dict(nmf_kwargs), gpu_kwargs))
        k = int(nmf_kwargs["n_components"])
        return [(np.zeros((k, X.shape[1])), np.zeros((X.shape[0], k))) for _ in seeds]

    monkeypatch.setattr(gpu_mod, "_nmf_gpu_mu", fake_mu)
    gpu_kwargs = {"device": "cpu", "dtype": "fp64"}
    gpu_mod.configure_nmf_engine(mock_cnmf, engine="gpu", gpu_kwargs=gpu_kwargs)

    mock_cnmf.factorize(worker_i=0, total_workers=1)

    assert len(captured) == 2                          # n_iter=2 replicates, batch=1 -> 2 launches
    replicate_params = load_df_from_npz(mock_cnmf.paths["nmf_replicate_parameters"])
    expected_seeds = set(replicate_params["nmf_seed"])
    observed_seeds = set()
    for seeds, kw, gk in captured:
        assert len(seeds) == 1                         # default batch=1 -> one seed per launch
        assert kw["n_components"] == 5                 # set per k by factorize
        assert "beta_loss" in kw and "init" in kw      # original run params forwarded
        assert gk == gpu_kwargs                        # resolved GPU kwargs passed through
        observed_seeds.update(seeds)
    assert observed_seeds == expected_seeds            # exact seeds from prepared replicate params
    for iter_i in replicate_params["iter"]:
        assert os.path.exists(mock_cnmf.paths["iter_spectra"] % (5, iter_i))


def test_refit_usage_gpu_engine_passes_fixed_h_update_h_false_and_gpu_kwargs(mock_cnmf, monkeypatch, tmp_path):
    """cNMF refit_usage should route fixed-H consensus refits through the GPU adapter."""
    import cnmf.nmf_gpu as gpu_mod

    write_minimal_nmf_run_params(mock_cnmf)
    X = pd.DataFrame(
        np.arange(12, dtype=np.float64).reshape(4, 3) + 1,
        index=[f"cell{i}" for i in range(4)],
        columns=[f"gene{i}" for i in range(3)],
    )
    spectra = pd.DataFrame(
        [[1.0, 0.3, 0.6], [0.2, 1.1, 0.4]],
        index=["program_a", "program_b"],
        columns=X.columns,
    )
    captured = []

    def fake_nmf_gpu(self, X_arg, nmf_kwargs):
        captured.append((X_arg, dict(nmf_kwargs)))
        return fake_gpu_nmf_output(X_arg, nmf_kwargs)

    monkeypatch.setattr(gpu_mod, "nmf_gpu", fake_nmf_gpu)
    gpu_kwargs = {"device": "cpu", "dtype": "fp64"}
    gpu_mod.configure_nmf_engine(mock_cnmf, engine="gpu", gpu_kwargs=gpu_kwargs)

    usages = mock_cnmf.refit_usage(X, spectra)

    assert len(captured) == 1
    X_arg, kw = captured[0]
    assert X_arg is X
    assert kw["n_components"] == 2
    assert np.allclose(kw["H"], spectra.values)
    assert kw["update_H"] is False
    assert kw["engine"] == "gpu"
    assert kw["gpu"] == gpu_kwargs
    assert "beta_loss" in kw and "init" in kw
    assert list(usages.index) == list(X.index)
    assert list(usages.columns) == list(spectra.index)
    assert usages.shape == (4, 2)


def test_refit_spectra_gpu_engine_routes_through_transposed_refit_usage(mock_cnmf, monkeypatch, tmp_path):
    """cNMF refit_spectra should use the same GPU fixed-H path through transposed refit_usage."""
    import cnmf.nmf_gpu as gpu_mod

    write_minimal_nmf_run_params(mock_cnmf)
    X = pd.DataFrame(
        np.arange(12, dtype=np.float64).reshape(4, 3) + 1,
        index=[f"cell{i}" for i in range(4)],
        columns=[f"gene{i}" for i in range(3)],
    )
    usage = pd.DataFrame(
        [[1.0, 0.2], [0.8, 0.4], [0.6, 0.7], [0.4, 1.0]],
        index=X.index,
        columns=["program_a", "program_b"],
    )
    captured = []

    def fake_nmf_gpu(self, X_arg, nmf_kwargs):
        captured.append((X_arg, dict(nmf_kwargs)))
        return fake_gpu_nmf_output(X_arg, nmf_kwargs)

    monkeypatch.setattr(gpu_mod, "nmf_gpu", fake_nmf_gpu)
    gpu_mod.configure_nmf_engine(mock_cnmf, engine="gpu", gpu_kwargs={"device": "cpu", "dtype": "fp64"})

    spectra = mock_cnmf.refit_spectra(X, usage)

    assert len(captured) == 1
    X_arg, kw = captured[0]
    assert X_arg.shape == (3, 4)                       # genes x cells after transpose
    assert np.allclose(kw["H"], usage.T.values)        # programs x cells fixed H
    assert kw["update_H"] is False
    assert kw["n_components"] == usage.shape[1]
    assert list(spectra.index) == list(usage.columns)
    assert list(spectra.columns) == list(X.columns)
    assert spectra.shape == (2, 3)


def test_consensus_gpu_engine_smoke_writes_expected_outputs(mock_cnmf, monkeypatch, tmp_path):
    """A tiny CPU-backed GPU-engine consensus run should write the expected consensus outputs."""
    import cnmf.nmf_gpu as gpu_mod

    counts_fn = generate_positive_counts_file(tmp_path)
    mock_cnmf.prepare(counts_fn, components=[2], n_iter=3, densify=True,
                      seed=14, num_highvar_genes=6, max_NMF_iter=5)

    norm_counts = sc.read(mock_cnmf.paths["normalized_counts"])
    genes = list(norm_counts.var.index)
    rng = np.random.default_rng(SEED)
    prototypes = rng.random((2, len(genes))) + 0.2
    merged = pd.DataFrame(
        np.vstack([
            prototypes[0] * 0.98,
            prototypes[0],
            prototypes[0] * 1.02,
            prototypes[1] * 0.98,
            prototypes[1],
            prototypes[1] * 1.02,
        ]),
        index=[f"iter{i}_topic{j}" for i in range(3) for j in range(1, 3)],
        columns=genes,
    )
    save_df_to_npz(merged, mock_cnmf.paths["merged_spectra"] % 2)

    def fake_nmf_gpu(self, X_arg, nmf_kwargs):
        return fake_gpu_nmf_output(X_arg, nmf_kwargs)

    monkeypatch.setattr(gpu_mod, "nmf_gpu", fake_nmf_gpu)
    gpu_mod.configure_nmf_engine(mock_cnmf, engine="gpu", gpu_kwargs={"device": "cpu", "dtype": "fp64"})

    mock_cnmf.consensus(k=2, density_threshold=2.0, local_neighborhood_size=0.5,
                        show_clustering=False, refit_usage=False)

    density = "2_0"
    expected_files = [
        mock_cnmf.paths["consensus_spectra"] % (2, density),
        mock_cnmf.paths["consensus_usages"] % (2, density),
        mock_cnmf.paths["gene_spectra_tpm"] % (2, density),
        mock_cnmf.paths["gene_spectra_score"] % (2, density),
        mock_cnmf.paths["starcat_spectra"] % (2, density),
    ]
    for path in expected_files:
        assert os.path.exists(path), f"Expected consensus output {path} not found."

    assert load_df_from_npz(mock_cnmf.paths["consensus_spectra"] % (2, density)).shape == (2, len(genes))
    assert load_df_from_npz(mock_cnmf.paths["consensus_usages"] % (2, density)).shape == (norm_counts.n_obs, 2)


def test_consensus_default_cpu_engine_keeps_original_sklearn_nmf_path(mock_cnmf, monkeypatch, tmp_path):
    """Consensus without GPU engine should not inject engine/gpu kwargs into sklearn NMF calls."""
    write_minimal_nmf_run_params(mock_cnmf)
    X = pd.DataFrame(
        np.arange(12, dtype=np.float64).reshape(4, 3) + 1,
        index=[f"cell{i}" for i in range(4)],
        columns=[f"gene{i}" for i in range(3)],
    )
    spectra = pd.DataFrame(
        [[1.0, 0.3, 0.6], [0.2, 1.1, 0.4]],
        index=["program_a", "program_b"],
        columns=X.columns,
    )
    captured = []

    def fake_cpu_nmf(X_arg, nmf_kwargs):
        captured.append(dict(nmf_kwargs))
        return fake_gpu_nmf_output(X_arg, nmf_kwargs)

    monkeypatch.setattr(mock_cnmf, "_nmf", fake_cpu_nmf)

    usages = mock_cnmf.refit_usage(X, spectra)

    assert len(captured) == 1
    kw = captured[0]
    assert "engine" not in kw
    assert "gpu" not in kw
    assert kw["update_H"] is False
    assert np.allclose(kw["H"], spectra.values)
    assert usages.shape == (4, 2)

"""Optional PyTorch NMF engine for cNMF.

This package module owns solver routing and cNMF integration. Shared runtime
helpers and the MU/CD implementations remain in focused sibling modules.
"""

import functools
from collections import OrderedDict

import numpy as np

from . import solver_cd, solver_mu, utils


__all__ = [
    "configure_nmf_engine",
    "factorize_gpu",
    "prepare_gpu",
    "solver_cd",
    "solver_mu",
    "utils",
]


_GPU_SOLVERS = {
    "mu": solver_mu._nmf_gpu_mu,
    "cd": solver_cd._nmf_gpu_cd,
}


def _nmf_gpu_batch(X, seeds, nmf_kwargs, gpu_kwargs=None):
    """Dispatch one same-k replicate batch to the selected GPU solver."""
    solver_name = str(
        nmf_kwargs.get("solver", utils.DEFAULT_NMF["solver"])
    ).strip().lower()
    try:
        solver = _GPU_SOLVERS[solver_name]
    except KeyError as exc:
        available = ", ".join(sorted(_GPU_SOLVERS))
        raise ValueError(
            f"GPU NMF solver {solver_name!r} is not available; "
            f"available solvers: {available}"
        ) from exc
    return solver(X, seeds, nmf_kwargs, gpu_kwargs)


def _nmf_gpu(args, X, nmf_kwargs, gpu_kwargs=None):
    """Single-replicate NMF adapter over the batch gateway."""
    gpu_kwargs = utils.gpu_kwargs_from_args(args)
    (result,) = _nmf_gpu_batch(
        X, [nmf_kwargs.get("random_state")], nmf_kwargs, gpu_kwargs
    )
    return result


def configure_nmf_engine(cnmf_constructor, args):
    """Construct and configure a cNMF instance for the selected execution engine."""
    engine = getattr(args, "engine", "cpu")
    if engine not in ("cpu", "gpu"):
        raise ValueError("engine must be 'cpu' or 'gpu'")

    if engine == "gpu":
        utils._validate_engine_args(args, _GPU_SOLVERS)

    cnmf_obj = cnmf_constructor(output_dir=args.output_dir, name=args.name)
    if engine == "cpu":
        return cnmf_obj

    # patch cNMF to use GPU NMF engine
    cnmf_obj._nmf = functools.partial(_nmf_gpu, args)
    original_prepare = cnmf_obj.prepare
    cnmf_obj.prepare = functools.partial(prepare_gpu, cnmf_obj, args, original_prepare)
    cnmf_obj.factorize = functools.partial(factorize_gpu, cnmf_obj, args)

    return cnmf_obj


def prepare_gpu(cnmf_obj, args, original_prepare, *prepare_args, **prepare_kwargs):
    """Prepare an unchanged cNMF run, then persist its explicit solver choice.

    Upstream ``cNMF.prepare`` derives the solver from ``beta_loss``. This
    adapter keeps all matrix preparation in upstream cNMF and changes only the
    saved factorization configuration used by factorize, resume, and consensus.
    """
    import yaml

    result = original_prepare(*prepare_args, **prepare_kwargs)

    config_path = cnmf_obj.paths["nmf_run_parameters"]
    with open(config_path, encoding="utf-8") as stream:
        run_parameters = yaml.safe_load(stream)
    if not isinstance(run_parameters, dict):
        raise ValueError(f"invalid NMF run configuration: {config_path}")
    run_parameters["solver"] = args.solver
    with open(config_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(run_parameters, stream, sort_keys=False)

    return result


def factorize_gpu(
    cnmf_obj,
    args,
    worker_i=0,
    total_workers=1,
    skip_completed_runs=False,
):
    """GPU ``factorize`` drop-in that batches same-k replicate seeds."""
    import pandas as pd
    import scanpy as sc
    import yaml

    try:
        from ..cnmf import load_df_from_npz, save_df_to_npz, worker_filter
    except ImportError:
        from cnmf import load_df_from_npz, save_df_to_npz, worker_filter

    gpu_kwargs = utils.gpu_kwargs_from_args(args)
    batch = utils._resolve_gpu_opts(gpu_kwargs)["batch"]
    run_params = load_df_from_npz(cnmf_obj.paths["nmf_replicate_parameters"])
    norm_counts = sc.read(cnmf_obj.paths["normalized_counts"])
    with open(cnmf_obj.paths["nmf_run_parameters"], encoding="utf-8") as stream:
        base_kwargs = yaml.load(stream, Loader=yaml.FullLoader)

    if skip_completed_runs:
        pending = run_params.index[run_params["completed"] == False]
        job_idx = worker_filter(pending, worker_i, total_workers)
    else:
        job_idx = worker_filter(range(len(run_params)), worker_i, total_workers)

    genes = norm_counts.var.index
    X_dense = (
        norm_counts.X.toarray()
        if hasattr(norm_counts.X, "toarray")
        else np.asarray(norm_counts.X)
    )
    by_k = OrderedDict()
    for idx in job_idx:
        params = run_params.iloc[idx, :]
        by_k.setdefault(int(params["n_components"]), []).append(
            (int(params["iter"]), int(params["nmf_seed"]))
        )

    for k, jobs in by_k.items():
        run_kwargs = dict(base_kwargs)
        run_kwargs["n_components"] = k
        for start in range(0, len(jobs), batch):
            chunk = jobs[start:start + batch]
            iters = [iteration for iteration, _seed in chunk]
            seeds = [seed for _iteration, seed in chunk]
            print(
                "[Worker %d]. k=%d: launching %d replicate(s), iters=%s."
                % (worker_i, k, len(chunk), iters)
            )
            results = _nmf_gpu_batch(X_dense, seeds, run_kwargs, gpu_kwargs)
            for (spectra, _usages), iteration in zip(results, iters):
                spectra = pd.DataFrame(
                    spectra,
                    index=np.arange(1, k + 1),
                    columns=genes,
                )
                save_df_to_npz(
                    spectra,
                    cnmf_obj.paths["iter_spectra"] % (k, iteration),
                )

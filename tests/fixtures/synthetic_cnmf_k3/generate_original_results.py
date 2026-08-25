#!/usr/bin/env python3
"""Generate stage-separated reference outputs with a pinned original cNMF source."""

from argparse import ArgumentParser
import hashlib
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys
from time import perf_counter

import pandas as pd


FIXTURE_DIR = Path(__file__).resolve().parent
DEFAULT_SPEC = FIXTURE_DIR / "run_spec.json"
DEFAULT_QC_SCRIPT = (
    FIXTURE_DIR
    / "reference_scripts"
    / "sc_blipper_fba3745"
    / "run_qc_cnmf.py"
)


def load_spec(path=DEFAULT_SPEC):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def load_upstream_cnmf(source_dir, expected_sha, expected_version):
    """Import cNMF only after verifying the checked-out upstream commit."""
    source_dir = Path(source_dir).resolve()
    actual_sha = subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"upstream cNMF checkout is {actual_sha}; expected {expected_sha}"
        )

    sys.path.insert(0, str(source_dir / "src"))
    from cnmf import __version__ as cnmf_version
    from cnmf import cNMF

    if cnmf_version != expected_version:
        raise RuntimeError(
            f"upstream cNMF version is {cnmf_version}; expected {expected_version}"
        )
    return cNMF, cnmf_version


def files_under(path):
    return {candidate for candidate in Path(path).rglob("*") if candidate.is_file()}


def snapshot_new_files(run_dir, before, stage_dir, name):
    """Copy files created by one cNMF stage while preserving run-relative paths."""
    created = sorted(files_under(run_dir) - before)
    destination_root = Path(stage_dir) / name
    for source in created:
        destination = destination_root / source.relative_to(run_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return created


def timed_stage(name, function, run_dir, result_dir, fixture_name):
    before = files_under(run_dir)
    started = perf_counter()
    function()
    elapsed = perf_counter() - started
    created = snapshot_new_files(
        run_dir,
        before,
        Path(result_dir) / name,
        fixture_name,
    )
    return elapsed, created


def factorize_with_iteration_tracking(cnmf_obj):
    """Run unmodified cNMF factorization while observing sklearn's n_iter result."""
    cnmf_module = importlib.import_module("cnmf.cnmf")
    original_factorization = cnmf_module.non_negative_factorization
    records = []

    def recording_factorization(*args, **kwargs):
        usages, spectra, n_iter = original_factorization(*args, **kwargs)
        records.append(
            {
                "replicate": len(records),
                "nmf_seed": int(kwargs["random_state"]),
                "n_components": int(kwargs["n_components"]),
                "n_iter": int(n_iter),
            }
        )
        return usages, spectra, n_iter

    cnmf_module.non_negative_factorization = recording_factorization
    try:
        cnmf_obj.factorize(worker_i=0, total_workers=1)
    finally:
        cnmf_module.non_negative_factorization = original_factorization
    return records


def generate(input_dir, work_dir, result_dir, source_dir, spec_path, standalone_qc_script):
    spec = load_spec(spec_path)
    fixture = spec["fixture"]
    nmf = spec["nmf"]
    upstream = spec["upstream"]
    name = fixture["name"]
    k = nmf["k"]
    n_replicates = nmf["replicates"]
    seed = fixture["seed"]
    max_nmf_iter = nmf["max_iter"]
    density_threshold = nmf["density_threshold"]
    local_neighborhood_size = nmf["local_neighborhood_size"]
    source_dir = Path(source_dir).resolve()
    cnmf_class, cnmf_version = load_upstream_cnmf(
        source_dir,
        upstream["commit"],
        upstream["version"],
    )

    input_dir = Path(input_dir).resolve()
    work_dir = Path(work_dir).resolve()
    result_dir = Path(result_dir).resolve()
    standalone_qc_script = Path(standalone_qc_script).resolve()
    qc_script_sha = hashlib.sha256(standalone_qc_script.read_bytes()).hexdigest()
    expected_qc_script_sha = spec["qc_reference"]["sha256"]
    if qc_script_sha != expected_qc_script_sha:
        raise RuntimeError(
            f"standalone QC script SHA256 is {qc_script_sha}; "
            f"expected {expected_qc_script_sha}"
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    cnmf_obj = cnmf_class(output_dir=str(work_dir), name=name)
    run_dir = work_dir / name
    timings = {}
    stage_files = {}

    timings["prepare"], stage_files["prepare"] = timed_stage(
        "prepare",
        lambda: cnmf_obj.prepare(
            counts_fn=str(input_dir / "synthetic_counts.h5ad"),
            components=[k],
            n_iter=n_replicates,
            seed=seed,
            genes_file=str(input_dir / "genes.txt"),
            beta_loss=nmf["beta_loss"],
            init=nmf["init"],
            max_NMF_iter=max_nmf_iter,
        ),
        run_dir,
        result_dir,
        name,
    )
    factorization_iterations = []

    def factorize():
        factorization_iterations.extend(factorize_with_iteration_tracking(cnmf_obj))

    timings["factorize"], stage_files["factorize"] = timed_stage(
        "factorize",
        factorize,
        run_dir,
        result_dir,
        name,
    )
    factorization_iterations_path = (
        result_dir / "factorize" / name / "factorization_iterations.tsv"
    )
    pd.DataFrame(factorization_iterations).to_csv(
        factorization_iterations_path,
        sep="\t",
        index=False,
    )
    stage_files["factorize"].append(factorization_iterations_path)
    timings["combine"], stage_files["combine"] = timed_stage(
        "combine",
        lambda: cnmf_obj.combine(components=[k]),
        run_dir,
        result_dir,
        name,
    )
    timings["consensus"], stage_files["consensus"] = timed_stage(
        "consensus",
        lambda: cnmf_obj.consensus(
            k=k,
            density_threshold=density_threshold,
            local_neighborhood_size=local_neighborhood_size,
            show_clustering=True,
            close_clustergram_fig=True,
        ),
        run_dir,
        result_dir,
        name,
    )

    standalone_qc_dir = result_dir / "standalone_qc" / name
    standalone_qc_dir.mkdir(parents=True, exist_ok=True)
    standalone_qc_output = standalone_qc_dir / name
    started = perf_counter()
    qc_environment = dict(os.environ)
    qc_environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(source_dir / "src"), qc_environment.get("PYTHONPATH")),
        )
    )
    subprocess.run(
        [
            sys.executable,
            str(standalone_qc_script),
            "--spectra",
            cnmf_obj.paths["merged_spectra"] % k,
            "--density",
            cnmf_obj.paths["local_density_cache"] % k,
            "--norm_counts",
            cnmf_obj.paths["normalized_counts"],
            "--params",
            cnmf_obj.paths["nmf_run_parameters"],
            "--output",
            str(standalone_qc_output),
            "--density_threshold",
            str(density_threshold),
            "--k",
            str(k),
        ],
        check=True,
        env=qc_environment,
    )
    timings["standalone_qc"] = perf_counter() - started
    stage_files["standalone_qc"] = sorted(files_under(standalone_qc_dir))

    manifest = {
        "source_repository": upstream["repository"],
        "source_sha": upstream["commit"],
        "standalone_qc_script": {
            "file": standalone_qc_script.name,
            "sha256": qc_script_sha,
        },
        "cnmf_version": cnmf_version,
        "dependencies": {
            package: version(package)
            for package in ("anndata", "numpy", "pandas", "scanpy", "scikit-learn", "scipy")
        },
        "configuration": {
            "name": name,
            "k": k,
            "replicates": n_replicates,
            "seed": seed,
            "solver": nmf["solver"],
            "beta_loss": nmf["beta_loss"],
            "init": nmf["init"],
            "max_nmf_iter": max_nmf_iter,
            "density_threshold": density_threshold,
            "local_neighborhood_size": local_neighborhood_size,
        },
        "factorization_iterations": {
            "minimum": min(record["n_iter"] for record in factorization_iterations),
            "median": median(record["n_iter"] for record in factorization_iterations),
            "maximum": max(record["n_iter"] for record in factorization_iterations),
            "reached_cap": sum(
                record["n_iter"] >= max_nmf_iter
                for record in factorization_iterations
            ),
            "all_converged_before_cap": all(
                record["n_iter"] < max_nmf_iter
                for record in factorization_iterations
            ),
        },
        "stages": {
            stage: {
                "elapsed_seconds": timings[stage],
                "file_count": len(stage_files[stage]),
            }
            for stage in ("prepare", "factorize", "combine", "consensus", "standalone_qc")
        },
    }
    with open(result_dir / "run_manifest.json", "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--standalone-qc-script",
        type=Path,
        default=DEFAULT_QC_SCRIPT,
    )
    args = parser.parse_args()
    generate(
        input_dir=args.input_dir,
        work_dir=args.work_dir,
        result_dir=args.result_dir,
        source_dir=args.source_dir,
        spec_path=args.spec,
        standalone_qc_script=args.standalone_qc_script,
    )


if __name__ == "__main__":
    main()

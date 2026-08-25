"""Runtime-generated regression tests against a pinned upstream cNMF release."""

import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import yaml

from cnmf import cNMF, load_df_from_npz
from cnmf.gpunmf import configure_nmf_engine


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "synthetic_cnmf_k3"
SPEC_PATH = FIXTURE_ROOT / "run_spec.json"
GENERATOR = FIXTURE_ROOT / "generate_synthetic_dataset.py"
REFERENCE_GENERATOR = FIXTURE_ROOT / "generate_original_results.py"

with open(SPEC_PATH, encoding="utf-8") as stream:
    SPEC = json.load(stream)

NAME = SPEC["fixture"]["name"]
K = SPEC["nmf"]["k"]
N_REPLICATES = SPEC["nmf"]["replicates"]
SEED = SPEC["fixture"]["seed"]
MAX_NMF_ITER = SPEC["nmf"]["max_iter"]
DENSITY_THRESHOLD = SPEC["nmf"]["density_threshold"]
LOCAL_NEIGHBORHOOD_SIZE = SPEC["nmf"]["local_neighborhood_size"]


@pytest.fixture(scope="module")
def generated_fixture(tmp_path_factory):
    """Generate deterministic input and pinned upstream outputs for this run."""
    upstream_source = os.environ.get("CNMF_UPSTREAM_SOURCE")
    if not upstream_source:
        pytest.skip(
            "set CNMF_UPSTREAM_SOURCE to the pinned dylkot/cNMF checkout "
            "to run synthetic regression tests"
        )

    upstream_source = Path(upstream_source).resolve()
    if not (upstream_source / ".git").exists():
        pytest.fail(f"CNMF_UPSTREAM_SOURCE is not a Git checkout: {upstream_source}")

    runtime_root = tmp_path_factory.mktemp("synthetic-cnmf-regression")
    input_root = runtime_root / "input"
    reference_root = runtime_root / "reference"
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--output-dir",
            str(input_root),
            "--spec",
            str(SPEC_PATH),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(REFERENCE_GENERATOR),
            "--input-dir",
            str(input_root),
            "--work-dir",
            str(runtime_root / "upstream-work"),
            "--result-dir",
            str(reference_root),
            "--source-dir",
            str(upstream_source),
            "--spec",
            str(SPEC_PATH),
        ],
        check=True,
    )

    with open(reference_root / "run_manifest.json", encoding="utf-8") as stream:
        manifest = json.load(stream)
    assert manifest["source_sha"] == SPEC["upstream"]["commit"]
    assert manifest["configuration"]["seed"] == SEED
    assert manifest["configuration"]["max_nmf_iter"] == MAX_NMF_ITER
    return SimpleNamespace(
        input_root=input_root,
        reference_root=reference_root,
        manifest=manifest,
    )


def _reference_cnmf(reference_root, stage):
    return cNMF(output_dir=str(reference_root / stage), name=NAME)


def _stage_reference(reference_root, stage, output_dir):
    """Copy one generated reference stage into a temporary cNMF run."""
    shutil.copytree(
        reference_root / stage / NAME,
        Path(output_dir) / NAME,
        dirs_exist_ok=True,
    )


def _dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _assert_adata_equal(actual_path, expected_path):
    actual = sc.read(actual_path)
    expected = sc.read(expected_path)
    pd.testing.assert_frame_equal(actual.obs, expected.obs)
    pd.testing.assert_frame_equal(actual.var, expected.var)
    np.testing.assert_allclose(
        _dense(actual.X),
        _dense(expected.X),
        rtol=1e-12,
        atol=1e-12,
    )


def _assert_npz_equal(actual_path, expected_path, rtol=1e-12, atol=1e-12):
    pd.testing.assert_frame_equal(
        load_df_from_npz(actual_path),
        load_df_from_npz(expected_path),
        check_dtype=False,
        rtol=rtol,
        atol=atol,
    )


def test_prepare_matches_dylkot_reference(generated_fixture, tmp_path):
    """Current prepare must reproduce the canonical dylkot/cNMF stage."""
    actual = cNMF(output_dir=str(tmp_path), name=NAME)
    expected = _reference_cnmf(generated_fixture.reference_root, "prepare")

    actual.prepare(
        counts_fn=str(generated_fixture.input_root / "synthetic_counts.h5ad"),
        components=[K],
        n_iter=N_REPLICATES,
        seed=SEED,
        genes_file=str(generated_fixture.input_root / "genes.txt"),
        beta_loss=SPEC["nmf"]["beta_loss"],
        init=SPEC["nmf"]["init"],
        max_NMF_iter=MAX_NMF_ITER,
    )

    for key in ("normalized_counts", "tpm"):
        _assert_adata_equal(actual.paths[key], expected.paths[key])
    for key in ("nmf_replicate_parameters", "tpm_stats"):
        _assert_npz_equal(actual.paths[key], expected.paths[key])
    assert Path(actual.paths["nmf_genes_list"]).read_text() == Path(
        expected.paths["nmf_genes_list"]
    ).read_text()
    with open(actual.paths["nmf_run_parameters"], encoding="utf-8") as stream:
        actual_parameters = yaml.safe_load(stream)
    with open(expected.paths["nmf_run_parameters"], encoding="utf-8") as stream:
        expected_parameters = yaml.safe_load(stream)
    assert actual_parameters == expected_parameters


def test_gpu_cd_factorize_matches_dylkot_cpu_reference(
    generated_fixture,
    tmp_path,
):
    """GPU-engine FP64 CD on CPU must reproduce canonical CPU-CD spectra."""
    pytest.importorskip("torch")

    _stage_reference(generated_fixture.reference_root, "prepare", tmp_path)
    args = SimpleNamespace(
        command="factorize",
        engine="gpu",
        solver="cd",
        beta_loss="frobenius",
        output_dir=str(tmp_path),
        name=NAME,
        gpu_device="cpu",
        gpu_dtype="fp64",
        gpu_allow_tf32=False,
        gpu_compile=False,
        gpu_eps=None,
        gpu_check_every=1,
        gpu_compile_block=None,
        gpu_batch=N_REPLICATES,
    )
    actual = configure_nmf_engine(cNMF, args)
    expected = _reference_cnmf(generated_fixture.reference_root, "factorize")

    actual.factorize(worker_i=0, total_workers=1)

    for replicate in range(N_REPLICATES):
        _assert_npz_equal(
            actual.paths["iter_spectra"] % (K, replicate),
            expected.paths["iter_spectra"] % (K, replicate),
            rtol=1e-10,
            atol=1e-12,
        )


def test_combine_matches_dylkot_reference(generated_fixture, tmp_path):
    """Current combine must reproduce the canonical merged spectra exactly."""
    _stage_reference(generated_fixture.reference_root, "prepare", tmp_path)
    _stage_reference(generated_fixture.reference_root, "factorize", tmp_path)
    actual = cNMF(output_dir=str(tmp_path), name=NAME)
    expected = _reference_cnmf(generated_fixture.reference_root, "combine")

    actual.combine(components=[K])

    _assert_npz_equal(
        actual.paths["merged_spectra"] % K,
        expected.paths["merged_spectra"] % K,
    )


@pytest.fixture(scope="module")
def consensus_case(generated_fixture, tmp_path_factory):
    """Run consensus once for the consensus and QC assertions."""
    output_dir = tmp_path_factory.mktemp("synthetic-regression-consensus")
    _stage_reference(generated_fixture.reference_root, "prepare", output_dir)
    _stage_reference(generated_fixture.reference_root, "combine", output_dir)
    actual = cNMF(output_dir=str(output_dir), name=NAME)
    actual.consensus(
        k=K,
        density_threshold=DENSITY_THRESHOLD,
        local_neighborhood_size=LOCAL_NEIGHBORHOOD_SIZE,
        show_clustering=False,
        close_clustergram_fig=True,
    )
    return actual, generated_fixture.reference_root


def test_consensus_matches_dylkot_reference(consensus_case):
    """Current consensus numerics must match canonical cNMF outputs."""
    actual, reference_root = consensus_case
    expected = _reference_cnmf(reference_root, "consensus")
    threshold = str(DENSITY_THRESHOLD).replace(".", "_")

    _assert_npz_equal(
        actual.paths["local_density_cache"] % K,
        expected.paths["local_density_cache"] % K,
    )
    for key in (
        "consensus_spectra",
        "consensus_usages",
        "gene_spectra_tpm",
        "gene_spectra_score",
        "starcat_spectra",
    ):
        _assert_npz_equal(
            actual.paths[key] % (K, threshold),
            expected.paths[key] % (K, threshold),
            rtol=1e-9,
            atol=1e-11,
        )


def test_consensus_qc_matches_run_qc_cnmf_reference(consensus_case):
    """Integrated QC must match shared columns from the original script."""
    actual, reference_root = consensus_case
    prefix = "k_3_dt_0_1"
    actual_prefix = Path(actual.output_dir) / NAME / NAME
    expected_prefix = reference_root / "standalone_qc" / NAME / NAME

    actual_annotation = pd.read_csv(
        f"{actual_prefix}.{prefix}.annotation.tsv",
        sep="\t",
    )
    expected_annotation = pd.read_csv(
        f"{expected_prefix}.{prefix}.annotation.tsv",
        sep="\t",
    )
    pd.testing.assert_frame_equal(
        actual_annotation,
        expected_annotation.loc[:, actual_annotation.columns],
        check_dtype=False,
        rtol=1e-9,
        atol=1e-11,
    )

    actual_edist = pd.read_csv(
        f"{actual_prefix}.{prefix}.edist.tsv",
        sep="\t",
        index_col=0,
    )
    expected_edist = pd.read_csv(
        f"{expected_prefix}.{prefix}.edist.tsv",
        sep="\t",
        index_col=0,
    )
    pd.testing.assert_frame_equal(
        actual_edist,
        expected_edist,
        check_dtype=False,
        rtol=1e-9,
        atol=1e-11,
    )
    assert Path(f"{actual_prefix}.{prefix}.edist.pdf").stat().st_size > 0

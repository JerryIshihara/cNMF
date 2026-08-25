import numpy as np
import pandas as pd

from cnmf.qc import edist, run_qc


def test_edistance_matches_squared_distance_definition():
    spectra = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 2.0],
        [0.0, 3.0],
    ])
    grouping = np.array([1, 1, 2, 2])

    actual = edist(spectra, grouping)

    np.testing.assert_allclose(
        actual.to_numpy(),
        np.array([[0.0, 12.0], [12.0, 0.0]]),
        rtol=0,
        atol=1e-12,
    )
    assert actual.index.tolist() == [1, 2]
    assert actual.columns.tolist() == [1, 2]


def test_run_qc_matches_run_qc_cnmf_reference_output(tmp_path):
    """Match the non-LOO QC output produced by sc-blipper's run_qc_cnmf.py."""
    # Reference source: ../sc-blipper/bin/run_qc_cnmf.py. This synthetic fixture
    # reproduces the state immediately after that script's clustering and fixed-H
    # usage refit: L2 spectra and cluster labels are fixed below, median spectra
    # are calculated with the script's groupby/median/row-normalization steps. A
    # fixed ``reorder`` Series represents the normalized usage totals returned by
    # the refit, in descending order. The expected annotation and E-distance
    # values pin the script's remaining non-LOO output logic. They are stored here
    # instead of executing a sibling sc-blipper checkout, keeping the cNMF test
    # self-contained in CI.
    inverse_sqrt_three = 1 / np.sqrt(3.0)
    l2_spectra = pd.DataFrame(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.6, 0.8, 0.0],
            [inverse_sqrt_three, inverse_sqrt_three, inverse_sqrt_three],
            [0.4, 0.5, np.sqrt(0.59)],
        ],
        index=[f"spectrum_{i}" for i in range(6)],
        columns=["gene_a", "gene_b", "gene_c"],
    )
    kmeans_cluster_labels = pd.Series(
        [1, 1, 2, 2, 3, 3],
        index=l2_spectra.index,
    )
    median_spectra = l2_spectra.groupby(kmeans_cluster_labels).median()
    median_spectra = median_spectra.div(median_spectra.sum(axis=1), axis=0)
    refit_usages = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
        ]
    )
    normalized_counts = refit_usages @ median_spectra.values
    local_density = pd.DataFrame(
        {"local_density": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]},
        index=l2_spectra.index,
    )

    # Fixed-H usage totals from the reference path, ordered by descending usage.
    reorder = pd.Series([2.7, 2.0, 1.3], index=[3, 1, 2])
    output = str(tmp_path / "reference")
    prefix = "k_3.dt_0_5"

    run_qc(
        output=output,
        prefix=prefix,
        k=3,
        reorder=reorder,
        median_spectra=median_spectra,
        l2_spectra=l2_spectra,
        local_density=local_density,
        kmeans_cluster_labels=kmeans_cluster_labels,
        normalized_counts=normalized_counts,
        refit_usages=refit_usages,
    )

    annotation_path = tmp_path / f"reference.{prefix}.annotation.tsv"
    edist_path = tmp_path / f"reference.{prefix}.edist.tsv"
    plot_path = tmp_path / f"reference.{prefix}.edist.pdf"

    annotation = pd.read_csv(annotation_path, sep="\t")
    expected_annotation = pd.DataFrame(
        {
            "gep": [1, 2, 3],
            "cluster": [3, 1, 2],
            "iter_count": [2, 2, 2],
            "iter_perc": [100.0, 100.0, 100.0],
            "k": [3, 3, 3],
            "min_edist": [0.9695920438537122, 1.12, 0.9695920438537122],
            "nonzero_genes": [3, 1, 2],
            "nonzero_perc": [100.0, 100 / 3, 200 / 3],
            "total_usage": [2.7, 2.0, 1.3],
            "run_iter_count": [6, 6, 6],
            "run_iter_count_perc": [100.0, 100.0, 100.0],
            "run_silhouette": [0.7354468523978769] * 3,
            "run_calinski_harabasz": [27.400170056360984] * 3,
            "run_davies_bouldin": [0.31744325019893016] * 3,
            "run_median_density": [0.35] * 3,
            "run_mean_density": [0.35] * 3,
            "run_r2": [1.0] * 3,
            "run_sse": [0.0] * 3,
            "run_tss": [3.3251911190193537] * 3,
        }
    )
    pd.testing.assert_frame_equal(
        annotation,
        expected_annotation,
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )

    spectra_edist = pd.read_csv(edist_path, sep="\t", index_col=0)
    expected_edist = pd.DataFrame(
        [
            [0.0, 1.971472259205413, 0.9695920438537122],
            [1.971472259205413, 0.0, 1.12],
            [0.9695920438537123, 1.12, 0.0],
        ],
        index=[1, 2, 3],
        columns=["1", "2", "3"],
    )
    pd.testing.assert_frame_equal(
        spectra_edist,
        expected_edist,
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0

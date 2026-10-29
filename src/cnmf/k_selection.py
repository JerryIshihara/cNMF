"""K-selection from combined HVG spectra, without full-gene consensus refits."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def k_selection_stats(cnmf_obj, k, norm_counts, streaming=False):
    """Calculate the original unfiltered stability and HVG reconstruction SSE.

    norm_counts must contain prepared, gene-scaled HVG expression in exactly
    the combined spectra's gene order. It is not raw counts or full-gene TPM.
    Streaming uses FP64 CPU CD refits; the cells-by-K usages remain in RAM.
    """
    from cnmf.cnmf import load_df_from_npz

    merged_spectra = load_df_from_npz(cnmf_obj.paths['merged_spectra'] % k)
    if not merged_spectra.columns.equals(norm_counts.var_names):
        raise ValueError('K-selection counts must match the combined spectra HVGs and gene order')
    expression = norm_counts.X
    if streaming and getattr(expression, 'format', None) == 'csc':
        raise ValueError('Streaming K-selection requires CSR or dense X, not CSC')

    # Match consensus(skip_density_and_return_after_stats=True): no density
    # filtering, the same KMeans seed, and L1-normalized cluster medians.
    l2_spectra = (merged_spectra.T / np.sqrt((merged_spectra**2).sum(axis=1))).T
    kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=1)
    kmeans_model.fit(l2_spectra)
    cluster_labels = pd.Series(kmeans_model.labels_ + 1, index=l2_spectra.index)
    median_spectra = l2_spectra.groupby(cluster_labels).median()
    median_spectra = (median_spectra.T / median_spectra.sum(axis=1)).T

    refit_options = dict(streaming=True) if streaming else {}
    usages = cnmf_obj.refit_usage(expression, median_spectra, **refit_options)
    usages = pd.DataFrame(usages, index=norm_counts.obs.index, columns=median_spectra.index)
    stability = silhouette_score(l2_spectra.values, cluster_labels, metric='euclidean')
    if streaming:
        from cnmf.streaming import reconstruction_stats_streaming
        _, prediction_error, _ = reconstruction_stats_streaming(
            expression, usages.values, median_spectra.values)
    else:
        prediction = usages.dot(median_spectra)
        expression = expression.todense() if sparse.issparse(expression) else expression
        prediction_error = ((expression - prediction)**2).sum().sum()

    # Retain the original stats schema and default threshold metadata. The
    # K-selection calculation itself does not apply a density filter.
    return pd.Series([k, 0.5, stability, prediction_error],
                     index=['k', 'local_density_threshold', 'silhouette', 'prediction_error'],
                     name='stats')


def k_selection_plot(cnmf_obj, close_fig=False, streaming=False, normalized_counts_fn=None):
    """Write the existing K-selection stats and plot, using one evaluation file.

    An explicit normalized_counts_fn lets a pooled run evaluate its combined
    spectra on a prepared sample without copying expression into the pooled
    run. Reuse that same file for every K; no scaling is recalculated here.
    """
    from cnmf.cnmf import load_df_from_npz, save_df_to_npz

    run_params = load_df_from_npz(cnmf_obj.paths['nmf_replicate_parameters'])
    if normalized_counts_fn is None:
        normalized_counts_fn = cnmf_obj.paths['normalized_counts']
    norm_counts = sc.read(normalized_counts_fn, backed='r' if streaming else None)
    try:
        stats = [k_selection_stats(cnmf_obj, k, norm_counts, streaming)
                 for k in sorted(set(run_params.n_components))]
    finally:
        if norm_counts.isbacked:
            norm_counts.file.close()
    stats = pd.DataFrame(stats).reset_index(drop=True)
    save_df_to_npz(stats, cnmf_obj.paths['k_selection_stats'])

    # Keep the original plot layout and output paths.
    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(stats.k, stats.silhouette, 'o-', color='b')
    ax1.set_ylabel('Stability', color='b', fontsize=15)
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    ax2.plot(stats.k, stats.prediction_error, 'o-', color='r')
    ax2.set_ylabel('Error', color='r', fontsize=15)
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    ax1.set_xlabel('Number of Components', fontsize=15)
    ax1.grid('on')
    plt.tight_layout()
    try:
        fig.savefig(cnmf_obj.paths['k_selection_plot'], dpi=250)
    finally:
        if close_fig:
            plt.close(fig)

#!/usr/bin/env python

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.preprocessing import scale
from tqdm import tqdm


def pairwise_dist(mat, grouping, dist='sqeuclidean', sample_correct=True, n_jobs=-1, verbose=True):
    """Average of pairwise distances between cells of each group.
    Adapted from: https://github.com/sanderlab/scPerturb/blob/master/package/src/scperturb/edistance.py
    
    Arguments
    ---------
    mat: :class:`~np.array`
        Data matrix.
    grouping: 'str' 
        Grouping variable.
    dist: `str` for any distance in scipy.spatial.distance (default: `sqeuclidean`)
        Distance metric to use in embedding space.
    sample_correct: `bool` (default: `True`)
        Whether make the estimator for sigma more unbiased (dividing by N-1 instead of N, similar to sample and population variance).
    n_jobs: `int` (default: `-1`)
        Number of jobs to use for parallelization. If `n_jobs=1`, no parallelization is used. The default uses all available threads.
    verbose: `bool` (default: `True`)
        Whether to show a progress bar iterating over all groups.

    Returns
    -------
    pwd: pandas.DataFrame
        DataFrame with average of pairwise PCA distances between all groups.
    """

    X = mat.copy()
    y = grouping.copy()
    groups = pd.unique(y)
    df = pd.DataFrame(index=groups, columns=groups, dtype=float)
    fct = tqdm if verbose else lambda x: x
    combis = list(combinations(groups, 2)) + [(x,x) for x in groups]
    def one_step(pair):
        p1, p2 = pair
        x1 = X[y==p1].copy()
        N = len(x1)
        x2 = X[y==p2].copy()
        pwd = pairwise_distances(x1, x2, metric=dist)
        M = len(x2)-1 if (p1==p2) & sample_correct else len(x2)
        factor = N * M
        mean_pwd = np.sum(pwd) / factor
        return (p1, p2, mean_pwd)
    res = Parallel(n_jobs=n_jobs)(delayed(one_step)(pair) for pair in fct(combis))
    for p1, p2, val in res:
        df.loc[p1, p2] = val
        df.loc[p2, p1] = val
    #df.index.name = obs_key
    #df.columns.name = obs_key
    #df.name = 'pairwise PCA distances'
    return df

def edist(mat, grouping, pwd=None, dist='sqeuclidean', sample_correct=True, n_jobs=1, verbose=True):
    """Computes the edistance to control. Accepts precomputed pwd.
    Computes the pairwise E-distances between all groups of rows defined in
    grouping. 

    Adapted from https://github.com/sanderlab/scPerturb/blob/master/package/src/scperturb/edistance.py
    Arguments
    ---------
    mat: :class:`~np.array`
        Data matrix.
    grouping: `str` 
        Array with grouping information. 
    dist: `str` for any distance in scipy.spatial.distance (default: `sqeuclidean`)
        Distance metric to use in embedding space.
    sample_correct: `bool` (default: `True`)
        Whether make the estimator for sigma more unbiased (dividing by N-1 instead of N, similar to sample and population variance).
    n_jobs: `int` (default: `-1`)
        Number of jobs to use for parallelization. If `n_jobs=1`, no parallelization is used. The default uses all available threads.
    verbose: `bool` (default: `True`)
        Whether to show a progress bar iterating over all groups.

    Returns
    -------
    estats: pandas.DataFrame
        DataFrame with pairwise E-distances between all groups.
    """
    pwd = pairwise_dist(mat, grouping, 
                                 dist=dist, sample_correct=sample_correct, 
                                 n_jobs=n_jobs, verbose=verbose) if pwd is None else pwd
    # derive basic statistics
    sigmas = np.diag(pwd)
    deltas = pwd
    estats = 2 * deltas - sigmas - sigmas[:, np.newaxis]
    return estats


def load_df_from_npz(filename):
    with np.load(filename, allow_pickle=True) as f:
        obj = pd.DataFrame(**f)
    return obj

def get_r2(X, U, H, log2p1=False, scale=False):
    if type(X) not in [np.ndarray]:
        raise ValueError("X must be a numpy array")
    if type(U) not in [np.ndarray]: 
        raise ValueError("U must be a numpy array")
    if type(H) not in [np.ndarray]:
        raise ValueError("H must be a numpy array")
            
    # Input matrix
    if X is None:        
        if log2p1:
            X = np.log2(X+1)
        if scale:
            X = scale(X, axis=0)

    # Reconstructed matrix
    X_pred = U.dot(H)
    if log2p1:
        X_pred = np.log2(X_pred+1)
    if scale:
        X_pred = scale(X_pred, axis=0)
        
    X_centered = X - np.mean(X)
    sse = np.sum((X - X_pred)**2)
    tss = np.sum(X_centered**2)
    r2 = 1 - (sse / tss)
    
    return r2, sse, tss


def plot_heatmap(df, outfile, scale_rows=False):
    # df: rows=sources, cols=conditions
    if scale_rows:
        df_plot = df.apply(
            lambda r: (r - r.mean()) / (r.std() if r.std() != 0 else 1),
            axis=1,
        )
    else:
        df_plot = df.copy()

    nrows, ncols = df_plot.shape
    side = 2 + (max(nrows, ncols) * 0.5)  # keep overall figure large enough
    fig, ax = plt.subplots(figsize=(side, side))

    sns.heatmap(
        df_plot,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        cbar_kws={"shrink": 0.5},
    )

    ax.set_box_aspect(1)  # force the heatmap panel to be square
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)



def nmf_variance_explained(X, U, H):
    """
    This is a crude way of estimating the variance explained by each program by leaving one out.
    
    :param X: Description
    :param U: Description
    :param H: Description
    """
    if sp.issparse(X):
        X=np.asarray(X.todense())
                    
    if type(X) not in [np.ndarray, np.matrix]:
        raise ValueError("X must be a numpy array")
    if type(U) not in [np.ndarray]: 
        raise ValueError("U must be a numpy array")
    if type(H) not in [np.ndarray]:
        raise ValueError("H must be a numpy array")
    
    k = H.shape[0]
    selectors = [np.arange(k) != i for i in range(k)]
    r2_base, sse_base, tss_base = get_r2(X, U, H)
    
    r2_wo = []
    sse_wo = []
    tss_wo = []
    for selector in selectors:
        r2, error, tss = get_r2(X, U[:,selector], H[selector,:])
        r2_wo.append(r2)
        sse_wo.append(error)
        tss_wo.append(tss)

    return r2_base, sse_base, tss_base, r2_wo, sse_wo, tss_wo


def run_qc(
    output,
    prefix,
    k,
    reorder,
    median_spectra,
    l2_spectra,
    local_density,
    kmeans_cluster_labels,
    normalized_counts,
    refit_usages,
):
    """Calculate and save QC using state already produced by ``consensus``."""
    # local_density contains all spectra, while l2_spectra has already been density-filtered.
    nruns = len(local_density)
    nruns_per_gep = nruns/k
    if sp.issparse(normalized_counts):
        normalized_counts = np.asarray(normalized_counts.todense())
    run_r2, run_sse, run_tss = get_r2(
        normalized_counts,
        refit_usages,
        median_spectra.values,
    )

    #-----------------------------------------------------------------------
    # Calculate how many iterations feed into each GEP    
    unique, counts = np.unique(kmeans_cluster_labels, return_counts=True)
    annotation = pd.DataFrame({
        'gep': None,
        'cluster': unique, 
        'iter_count': counts,
        'iter_perc': (counts/nruns_per_gep)*100,
        'k': k
    }).sort_values('cluster', ascending=True)

    #-----------------------------------------------------------------------
    print("Calculating e-distance metrics")
    
    # Calculate the edistance
    spectra_edist = edist(l2_spectra, kmeans_cluster_labels.to_numpy())
    
    # Re-order to match the clusters
    spectra_edist = spectra_edist.reindex(index=annotation['cluster']).reindex(columns=annotation['cluster'])
    
    # Calculate the minimal e-distance to other clusters for each cluster
    annotation['min_edist'] = spectra_edist.mask(np.eye(len(spectra_edist), dtype=bool)).min(axis=1).to_numpy()
   
    #-----------------------------------------------------------------------
    # Number of non-zero genes in spectra
    annotation['nonzero_genes'] =  (median_spectra != 0).sum(axis=1).to_numpy()
    annotation['nonzero_perc'] =  (annotation['nonzero_genes'] / median_spectra.shape[1]) *100
   
    #-----------------------------------------------------------------------
    # Re-order based on normalized usages, this is important to keep consistency 
    # with the main cNMF pipeline where GEPs are ordered by total usage across all cells.
    annotation['total_usage'] = annotation['cluster'].map(reorder)
 
    #-----------------------------------------------------------------------
    # Calculate clustering metrics, this is a sanity check to see if the clusters are well separated in the spectra space overall
    annotation['run_iter_count'] = l2_spectra.shape[0]
    annotation['run_iter_count_perc'] = (l2_spectra.shape[0]/nruns)*100 
    annotation['run_silhouette'] = silhouette_score(l2_spectra.values, kmeans_cluster_labels, metric='euclidean')
    annotation['run_calinski_harabasz'] = calinski_harabasz_score(l2_spectra.values, kmeans_cluster_labels)
    annotation['run_davies_bouldin'] = davies_bouldin_score(l2_spectra.values, kmeans_cluster_labels)
    annotation['run_median_density'] = np.median(local_density.iloc[:, 0])
    annotation['run_mean_density'] = np.mean(local_density.iloc[:, 0])
    annotation['run_r2'] = run_r2
    annotation['run_sse'] = run_sse
    annotation['run_tss'] = run_tss

    annotation.index = range(1, len(annotation)+1)
    annotation = annotation.loc[reorder.index,:]
    annotation['gep'] = range(1, len(annotation)+1)
    annotation.index = range(1, len(annotation)+1)
    
    # Re order edits
    spectra_edist = spectra_edist.loc[annotation['cluster'], annotation['cluster']]
    spectra_edist.index = range(1, len(annotation)+1)
    spectra_edist.columns = range(1, len(annotation)+1)

    # Save the results
    annotation.to_csv(f"{output}.{prefix}.annotation.tsv", sep="\t", index=False)
    spectra_edist.to_csv(f"{output}.{prefix}.edist.tsv", sep="\t", index=True)
    
    plot_heatmap(spectra_edist, f"{output}.{prefix}.edist.pdf", scale_rows=False)

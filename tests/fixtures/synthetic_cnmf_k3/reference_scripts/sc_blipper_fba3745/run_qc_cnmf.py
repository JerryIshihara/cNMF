#!/usr/bin/env python

import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import scanpy as sc
from cnmf import cNMF
import yaml
from sklearn.decomposition import non_negative_factorization
from sklearn.preprocessing import scale
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

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


if __name__ == "__main__":
              
    parser = argparse.ArgumentParser(description="Calculate QC metrics on an NMF run")
    parser.add_argument('--spectra', required=True, help='path to merged spectra npz')
    parser.add_argument('--density', required=True, help='path to local density npz')
    parser.add_argument('--norm_counts', required=True, help='path to norm_counts h5ad')
    parser.add_argument('--params', required=False, help='path to nmf params yaml (can be skipped if --skip_varexp is set)', default=None)
    parser.add_argument('--density_threshold', required=True, type=float, help='density threshold (float)')
    parser.add_argument('--output', required=True, help='output prefix')
    parser.add_argument('--k', required=True, type=int, help='number of clusters')
    parser.add_argument('--skip_varexp', dest='skip_varexp', action='store_true', help='Skip calculating variance explained (default: false)')
    parser.set_defaults(skip_varexp=False)
    params = parser.parse_args()

    # This script is partially reverse engineered from cNMF
    # https://github.com/dylkot/cNMF/blob/main/src/cnmf/cnmf.py
    
    #---------------------------------------------
    print("Re-clustering spectra")
    
    # Re-construct the clustering cNMF does
    merged_spectra = load_df_from_npz(params.spectra)
    
    # L2 normalize the raw spectra
    l2_spectra = (merged_spectra.T/np.sqrt((merged_spectra**2).sum(axis=1))).T
    
    # Filter for density
    local_density = load_df_from_npz(params.density)
    
    nruns = l2_spectra.shape[0]
    nruns_per_gep = nruns/params.k
    
    density_filter = local_density.iloc[:, 0] < params.density_threshold
    l2_spectra = l2_spectra.loc[density_filter, :]
    if l2_spectra.shape[0] == 0:
        raise RuntimeError("Zero components remain after density filtering. Consider increasing density threshold")

    # Cluster
    kmeans_model = KMeans(n_clusters=params.k, n_init=10, random_state=1)
    kmeans_model.fit(l2_spectra)
    kmeans_cluster_labels = pd.Series(kmeans_model.labels_+1, index=l2_spectra.index)
    kmeans_cluster_labels = kmeans_cluster_labels
    
    # Find median usage for each gene across cluster
    # Row index is the cluster labels
    median_spectra = l2_spectra.groupby(kmeans_cluster_labels).median()

    # Normalize median spectra to probability distributions.
    median_spectra = (median_spectra.T/median_spectra.sum(1)).T
    
    #-----------------------------------------------------------------------
    # Calculate how many iterations feed into each GEP    
    unique, counts = np.unique(kmeans_cluster_labels, return_counts=True)
    annotation = pd.DataFrame({
        'gep': None,
        'cluster': unique, 
        'iter_count': counts,
        'iter_perc': (counts/nruns_per_gep)*100,
        'k': params.k
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
    # This is needed to be able to properly order the GEPs by total usage across all cells,
    # which is important for consistency with the main cNMF pipeline.
    # Secondly its also needed to estimate the error metrics.
    print("Re-fitting NMF with fixed H to get usages")
    
    norm_counts = sc.read(params.norm_counts)
    refit_nmf_kwargs = yaml.load(open(params.params), Loader=yaml.FullLoader)
    
    if type(median_spectra) is pd.DataFrame:
        refit_nmf_kwargs.update(dict(n_components = median_spectra.shape[0], H = median_spectra.values, update_H = False))
    else:
        refit_nmf_kwargs.update(dict(n_components = median_spectra.shape[0], H = median_spectra, update_H = False))
                
    (usages, _, niter) = non_negative_factorization(norm_counts.X,  **refit_nmf_kwargs)
    rf_usages = pd.DataFrame(usages, index=norm_counts.obs.index, columns=median_spectra.index)  
    
    #-----------------------------------------------------------------------
    # Re-order based on normalized usages, this is important to keep consistency 
    # with the main cNMF pipeline where GEPs are ordered by total usage across all cells.
    norm_usages = rf_usages.div(rf_usages.sum(axis=1), axis=0)    
    annotation['total_usage'] = norm_usages.values.sum(axis=0)
    reorder = norm_usages.sum(axis=0).sort_values(ascending=False)

    #-----------------------------------------------------------------------
    # Calculate the LOO variance explained     
    # Read the 'counts' data that the nmf was fit on
    if not params.skip_varexp:
        
        print("Estimating variance explained per program by leaving one out")
        # Baseline r2
        print(type(norm_counts.X))
        r2_base, sse_base, tss_base, r2_wo, sse_wo, tss_wo = nmf_variance_explained(norm_counts.X, rf_usages.values, median_spectra.values)
    
        annotation['r2_diff'] = r2_base - np.array(r2_wo)
        annotation['loo_r2'] = r2_wo
        annotation['loo_sse'] = sse_wo
        annotation['loo_tss'] = tss_wo
        annotation['run_r2'] = r2_base
        annotation['run_sse'] = sse_base
        annotation['run_tss'] = tss_base
 
    #-----------------------------------------------------------------------
    # Calculate clustering metrics, this is a sanity check to see if the clusters are well separated in the spectra space overall
    annotation['run_iter_count'] = l2_spectra.shape[0]
    annotation['run_iter_count_perc'] = (l2_spectra.shape[0]/nruns)*100 
    annotation['run_silhouette'] = silhouette_score(l2_spectra.values, kmeans_cluster_labels, metric='euclidean')
    annotation['run_calinski_harabasz'] = calinski_harabasz_score(l2_spectra.values, kmeans_cluster_labels)
    annotation['run_davies_bouldin'] = davies_bouldin_score(l2_spectra.values, kmeans_cluster_labels)
    annotation['run_median_density'] = np.median(local_density.iloc[:, 0])
    annotation['run_mean_density'] = np.mean(local_density.iloc[:, 0])

    annotation.index = range(1, len(annotation)+1)
    annotation = annotation.loc[reorder.index,:]
    annotation['gep'] = range(1, len(annotation)+1)
    annotation.index = range(1, len(annotation)+1)
    
    # Re order edits
    spectra_edist = spectra_edist.loc[annotation['cluster'], annotation['cluster']]
    spectra_edist.index = range(1, len(annotation)+1)
    spectra_edist.columns = range(1, len(annotation)+1)

    # Save the results
    prefix = f"k_{params.k}.dt_{params.density_threshold}".replace(".", "_")
    annotation.to_csv(f"{params.output}.{prefix}.annotation.tsv", sep="\t", index=False)
    spectra_edist.to_csv(f"{params.output}.{prefix}.edist.tsv", sep="\t", index=True)
    
    plot_heatmap(spectra_edist, f"{params.output}.{prefix}.edist.pdf", scale_rows=False)
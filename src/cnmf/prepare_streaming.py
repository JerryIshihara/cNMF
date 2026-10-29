"""Chunked prepare inputs; preserve cNMF paths without materializing expression."""
from contextlib import contextmanager, ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from . import streaming


@contextmanager
def _read_input(path):
    """Read only X and cell/gene annotations, not layers or embeddings."""
    if Path(path).suffix != '.h5ad':
        raise ValueError('Streaming prepare requires H5AD input')
    with h5py.File(path, 'r') as handle:
        x = handle['X']
        if isinstance(x, h5py.Group):
            x = ad.io.sparse_dataset(x)
            if x.format != 'csr':
                raise ValueError('Streaming prepare requires dense or CSR input, not CSC')
        yield x, ad.io.read_elem(handle['obs']), ad.io.read_elem(handle['var'])


def _write_h5ad(path, obs, var, blocks, as_sparse):
    """Write dense/CSR X incrementally; annotations remain in memory."""
    ad.AnnData(obs=obs.copy(), var=var.copy()).write_h5ad(path)
    with h5py.File(path, 'r+') as handle:
        shape = (len(obs), len(var))
        values = None
        for start, stop, block in blocks:
            if as_sparse:
                block = sparse.csr_matrix(block)
                if values is None:
                    # Matching int64 indices/pointers permit >2**31 nonzeros
                    # and avoid sklearn's mixed-index-dtype failure.
                    block.indices = block.indices.astype(np.int64)
                    block.indptr = block.indptr.astype(np.int64)
                    ad.io.write_elem(handle, 'X', block)
                    values = ad.io.sparse_dataset(handle['X'])
                else:
                    values.append(block)
            else:
                if sparse.issparse(block):
                    block = block.toarray()
                if values is None:
                    values = handle.create_dataset('X', shape=shape, dtype=block.dtype)
                    values.attrs.update({'encoding-type': 'array', 'encoding-version': '0.2.0'})
                values[start:stop] = block


def _tpm_blocks(x, rows, densify):
    from .cnmf import compute_tpm
    for start, stop, block in streaming._blocks(x, rows, dtype=None):
        if densify and sparse.issparse(block):
            block = block.toarray()
        yield start, stop, compute_tpm(ad.AnnData(block)).X


def _scaled_blocks(x, columns, scales, rows, obs):
    for start, stop, block in streaming._blocks(x, rows):
        block = block[:, columns]
        if sparse.issparse(block):
            block = block.multiply(1 / scales).tocsr()
        else:
            block = block / scales
        zero = np.asarray(block.sum(axis=1) == 0).reshape(-1)
        if zero.any():
            raise ValueError('Cells have zero counts of overdispersed genes: %s' %
                             ', '.join(obs.index[start:stop][zero][:4]))
        yield start, stop, block


def prepare_inputs_streaming(counts_fn, paths, genes_file, tpm_fn=None, densify=False):
    """Write the usual TPM, statistics, HVG list and scaled-count files.

    Requires a precomputed shared HVG list. Gene scaling uses this input's
    cells, exactly as ordinary prepare; no shared sample scaling is introduced.
    Only X/obs/var are written, not unrelated layers, raw or embeddings.
    """
    from .cnmf import save_df_to_npz
    if genes_file is None:
        raise ValueError('Streaming prepare requires --genes-file; reuse the sampling HVG list')
    genes = Path(genes_file).read_text().rstrip().split('\n')
    if not genes or not all(genes) or len(genes) != len(set(genes)):
        raise ValueError('HVG list must contain unique, nonempty gene IDs')
    destinations = [Path(paths[key]).resolve() for key in
                    ('tpm', 'normalized_counts', 'tpm_stats', 'nmf_genes_list')]
    if Path(counts_fn).resolve() in destinations or (tpm_fn and Path(tpm_fn).resolve() in destinations):
        raise ValueError('Streaming prepare inputs must differ from its output paths')
    with TemporaryDirectory(prefix='.prepare-', dir=Path(paths['tpm']).parent) as folder:
        folder = Path(folder)
        with ExitStack() as stack:
            x, obs, var = stack.enter_context(_read_input(counts_fn))
            if min(x.shape) < 1 or x.shape[0] < 2:
                raise ValueError('Streaming prepare requires at least two cells and one gene')
            if not var.index.is_unique:
                raise ValueError('Count gene IDs must be unique')
            columns = var.index.get_indexer(genes)
            if (columns < 0).any():
                raise ValueError('Counts are missing HVG genes')
            rows = streaming._chunk_size(x.shape, 0)
            counts_sparse = getattr(x, 'format', None) == 'csr' and not densify
            if tpm_fn is None:
                _write_h5ad(folder/'tpm.h5ad', obs, var, _tpm_blocks(x, rows, densify), counts_sparse)
                tpm_path = folder/'tpm.h5ad'
            else:
                tpm_path = tpm_fn
            tpm, tpm_obs, tpm_var = stack.enter_context(_read_input(tpm_path))
            if not obs.index.equals(tpm_obs.index):
                raise ValueError('Counts and TPM cell IDs/order differ')
            if not tpm_var.index.is_unique or not pd.Index(genes).isin(tpm_var.index).all():
                raise ValueError('TPM must have unique gene IDs and contain every HVG')
            mean, variance = streaming.gene_stats_streaming(tpm)
            save_df_to_npz(pd.DataFrame({'__mean': mean, '__std': np.sqrt(variance)},
                                        index=tpm_var.index), folder/'tpm_stats.df.npz')
            # Counts scaling uses ddof=1; TPM statistics above use ddof=0.
            counts_mean, counts_variance = streaming.gene_stats_streaming(x)
            scales = np.sqrt(counts_variance[columns] * x.shape[0] / (x.shape[0]-1))
            scaled_var = var.iloc[columns].copy()
            if getattr(tpm, 'format', None) == 'csr':
                scales[scales == 0] = 1
                scaled_var['mean'] = counts_mean[columns]
                scaled_var['std'] = scales
            if not np.isfinite(scales).all() or (scales == 0).any():
                raise ValueError('HVG scaling requires finite nonzero standard deviations')
            _write_h5ad(folder/'norm_counts.h5ad', obs, scaled_var,
                         _scaled_blocks(x, columns, scales, rows, obs), counts_sparse)
            if tpm_fn is not None:
                _write_h5ad(folder/'tpm.h5ad', tpm_obs, tpm_var,
                             streaming._blocks(tpm, streaming._chunk_size(tpm.shape, 0), dtype=None),
                             getattr(tpm, 'format', None) == 'csr')
            (folder/'genes.txt').write_text('\n'.join(genes))
        for key, name in [('tpm', 'tpm.h5ad'), ('normalized_counts', 'norm_counts.h5ad'),
                          ('tpm_stats', 'tpm_stats.df.npz'), ('nmf_genes_list', 'genes.txt')]:
            (folder/name).replace(paths[key])

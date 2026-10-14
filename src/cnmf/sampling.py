"""Cell sampling that preserves input genes, counts, and annotations."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import h5py
import numpy as np
from scipy import sparse


def _sample_filenames(counts_fn, sample_times, sample_prefix=None):
    """Resolve the prefix and filenames for each sampling draw."""
    if sample_prefix is None:
        sample_prefix = Path(counts_fn).stem + '_sample'
    if (not sample_prefix or sample_prefix in ('.', '..')
            or Path(sample_prefix).name != sample_prefix):
        raise ValueError('sample_prefix must be a filename prefix without directories')

    filenames = []
    for number in range(1, sample_times + 1):
        prefix = '%s_%03d' % (sample_prefix, number)
        filenames.append({
            'sample_id': prefix,
            'counts': prefix + '.h5ad',
            'cell_indices': prefix + '.cell_indices.npy',
        })
    return sample_prefix, filenames


def _write_sample_matrix(group, name, matrix, indices, chunk_size, pairwise=False):
    """Stream a selected matrix in its sparse storage direction.

    Dense input becomes CSR without rounding or thresholding. Keep CSC input
    column-compressed: row indexing a backed CSC matrix can read the entire
    matrix. Its column blocks use the equivalent dense-element chunk budget.
    Use int64 pointers so exports can exceed 2**31 nonzero entries.
    """
    fmt = getattr(matrix, 'format', getattr(matrix, 'format_str', None))
    fmt = 'csc' if fmt == 'csc' else 'csr'
    n_cols = len(indices) if pairwise else matrix.shape[1]
    shape = (len(indices), n_cols)
    result = group.create_group(name)
    result.attrs.update({'encoding-type': fmt + '_matrix',
                         'encoding-version': '0.1.0', 'shape': shape})
    values = result.create_dataset('data', shape=(0,), maxshape=(None,),
                                   dtype=matrix.dtype)
    positions = result.create_dataset('indices', shape=(0,), maxshape=(None,),
                                      dtype='int64' if max(shape) >= 2**31 else 'int32')
    size = shape[1] if fmt == 'csc' else shape[0]
    pointers = result.create_dataset('indptr', shape=(size + 1,), dtype='int64')
    if fmt == 'csc':
        chunk_size = max(1, chunk_size * matrix.shape[1] // max(1, matrix.shape[0]))
    offset = 0
    for start in range(0, size, chunk_size):
        stop = min(start + chunk_size, size)
        if fmt == 'csc':
            columns = indices[start:stop] if pairwise else slice(start, stop)
            chunk = sparse.csc_matrix(matrix[:, columns][indices, :])
        else:
            chunk = matrix[indices[start:stop], :]
            if pairwise:
                chunk = chunk[:, indices]
            chunk = sparse.csr_matrix(chunk)
        end = offset + chunk.nnz
        values.resize((end,))
        positions.resize((end,))
        values[offset:end] = chunk.data
        positions[offset:end] = chunk.indices
        pointers[start + 1:stop + 1] = chunk.indptr[1:].astype(np.int64) + offset
        offset = end


def _sample_export(data, indices, filename, chunk_size):
    """Write one ordinary sparse H5AD without materializing the sampled X.

    X, layers, raw/X and pairwise cell matrices are streamed separately.
    Annotations already loaded by AnnData's backed reader stay in memory;
    this bounds export matrix buffers, not the entire sampling/prepare RAM.
    Publish the file only after every matrix has been written successfully.
    """
    filename = Path(filename)
    selected = data[indices]
    metadata = ad.AnnData(obs=selected.obs.copy(), var=selected.var.copy(),
                         obsm=dict(selected.obsm), varm=dict(selected.varm),
                         varp=dict(selected.varp), uns=dict(selected.uns))
    with TemporaryDirectory(prefix='.' + filename.stem + '-', dir=filename.parent) as folder:
        temporary = Path(folder) / filename.name
        metadata.write_h5ad(temporary)
        with h5py.File(temporary, 'a') as output:
            _write_sample_matrix(output, 'X', data.X, indices, chunk_size)
            for name, matrix in data.layers.items():
                _write_sample_matrix(output['layers'], name, matrix, indices, chunk_size)
            for name, matrix in data.obsp.items():
                _write_sample_matrix(output['obsp'], name, matrix, indices, chunk_size,
                                     pairwise=True)
            if data.raw is not None:
                raw = output.create_group('raw')
                # Gene annotations are unchanged: raw can contain more genes
                # than X, and these must not be intersected or reordered.
                with h5py.File(data.filename, 'r') as source:
                    raw.attrs.update(source['raw'].attrs)
                    for name in ('var', 'varm'):
                        if name in source['raw']:
                            source.copy(source['raw'][name], raw, name=name)
                _write_sample_matrix(raw, 'X', data.raw.X, indices, chunk_size)
        temporary.replace(filename)


def _sample_hvgs(data, output_dir, sample_prefix, sample_numgenes=2000,
                 sample_hvg_mode='auto', sample_hvg_chunk_size=2000):
    """Save one ordered full-input HVG list for every draw's prepare step.

    Reuse an existing mask in auto mode. Otherwise normalize temporary row
    chunks with cNMF's compute_tpm and accumulate population mean/variance,
    then apply the same Fano ranking used by cNMF's sparse HVG calculation.
    Neither the input counts nor their annotations are modified.
    """
    if not isinstance(sample_numgenes, int) or sample_numgenes < 0:
        raise ValueError('sample_numgenes must be a nonnegative integer')
    if sample_hvg_mode not in ('auto', 'recompute'):
        raise ValueError('sample_hvg_mode must be auto or recompute')
    if not isinstance(sample_hvg_chunk_size, int) or sample_hvg_chunk_size < 1:
        raise ValueError('sample_hvg_chunk_size must be a positive integer')
    if sample_numgenes == 0:
        return None

    info = {'mode': sample_hvg_mode, 'n_input_cells': data.n_obs}
    if sample_hvg_mode == 'auto' and 'highly_variable' in data.var:
        mask = data.var['highly_variable']
        if mask.isna().any() or mask.dtype.kind != 'b':
            raise ValueError('var["highly_variable"] must contain boolean values without missing entries')
        selected = mask.to_numpy(dtype=bool)
        # An existing mask does not establish its selection method/population.
        info.update(source='var/highly_variable', method=None)
        print('Reusing var["highly_variable"] without recalculating HVGs.')
    else:
        from sklearn.preprocessing import StandardScaler
        from .cnmf import compute_tpm, get_highvar_genes_from_stats

        print('Computing shared cNMF HVGs from all %d input cells (%d per chunk).'
              % (data.n_obs, sample_hvg_chunk_size))
        scaler = StandardScaler(with_mean=False)
        for start in range(0, data.n_obs, sample_hvg_chunk_size):
            chunk = data.X[start:start + sample_hvg_chunk_size]
            values = chunk.data if sparse.issparse(chunk) else chunk
            if not np.isfinite(values).all():
                raise ValueError('HVG input contains NaN or Inf')
            if (values < 0).any():
                raise ValueError('HVG selection requires nonnegative counts; use sample_numgenes=0 to skip')
            tpm = compute_tpm(ad.AnnData(chunk))
            scaler.partial_fit(tpm.X)
        stats, _ = get_highvar_genes_from_stats(
            scaler.mean_, scaler.var_, numgenes=sample_numgenes)
        selected = stats['high_var'].to_numpy()
        info.update(source='computed', method='cnmf_fano_ratio',
                    numgenes=sample_numgenes, chunk_size=sample_hvg_chunk_size,
                    normalization='CPM (target_sum=1e6)',
                    statistics_dtype='float64')

    genes = data.var_names[selected]
    if len(genes) == 0:
        raise ValueError('The shared HVG list is empty')
    filename = sample_prefix + '.overdispersed_genes.txt'
    genes_path = Path(output_dir) / filename
    genes_path.write_text('\n'.join(genes) + '\n')
    info.update(genes_file=filename, n_genes=len(genes))
    print('Shared HVGs saved; prepare each draw with --genes-file %s' % genes_path)
    return info


def _sample_embedding(data, output_dir, sample_pca_mode='auto',
                      sample_pca_components=30, sample_pca_chunk_size=2000,
                      sample_pca_input='counts'):
    """Reuse X_pca or fit one shared incremental PCA without changing counts.

    Newly computed coordinates are stored in a float32 NPY memmap, not added
    to the input H5AD. Fitting uses float64 batches in input cell order and all
    input genes; this does not select the HVGs used by downstream cNMF.
    """
    if sample_pca_mode not in ('auto', 'recompute'):
        raise ValueError('sample_pca_mode must be auto or recompute')
    if sample_pca_input not in ('counts', 'preprocessed'):
        raise ValueError('sample_pca_input must be counts or preprocessed')
    if not isinstance(sample_pca_chunk_size, int) or sample_pca_chunk_size < 1:
        raise ValueError('sample_pca_chunk_size must be a positive integer')

    if sample_pca_mode == 'auto' and 'X_pca' in data.obsm:
        embedding = np.asarray(data.obsm['X_pca'])
        if (embedding.ndim != 2 or embedding.shape[0] != data.n_obs
                or embedding.shape[1] < 1 or embedding.dtype.kind not in 'fiu'):
            raise ValueError('obsm["X_pca"] must be a real cells-by-components matrix')
        for start in range(0, data.n_obs, sample_pca_chunk_size):
            if not np.isfinite(embedding[start:start + sample_pca_chunk_size]).all():
                raise ValueError('obsm["X_pca"] contains NaN or Inf')
        print('Reusing obsm["X_pca"] with %d components.' % embedding.shape[1])
        return embedding, {
            'source': 'obsm/X_pca', 'mode': sample_pca_mode,
            'components': embedding.shape[1], 'dtype': str(embedding.dtype),
            # An existing embedding does not establish its preprocessing history.
            'preprocessing': None,
        }

    from sklearn.decomposition import IncrementalPCA

    if (not isinstance(sample_pca_components, int)
            or not 1 <= sample_pca_components <= min(data.n_obs - 1, data.n_vars)):
        raise ValueError('sample_pca_components must be between 1 and min(cells - 1, genes)')
    if sample_pca_chunk_size < sample_pca_components:
        raise ValueError('sample_pca_chunk_size must be at least sample_pca_components')

    def batches():
        start = 0
        while start < data.n_obs:
            end = min(start + sample_pca_chunk_size, data.n_obs)
            # Merge a short final batch so every fit has at least n_components rows.
            if data.n_obs - end < sample_pca_components:
                end = data.n_obs
            chunk = data.X[start:end]
            if sparse.issparse(chunk):
                chunk = chunk.astype(np.float64).toarray()
            else:
                chunk = np.array(chunk, dtype=np.float64, copy=True)
            if not np.isfinite(chunk).all():
                raise ValueError('PCA input contains NaN or Inf')
            if sample_pca_input == 'counts':
                if (chunk < 0).any():
                    raise ValueError('sample_pca_input=counts requires nonnegative values')
                totals = chunk.sum(axis=1, keepdims=True)
                totals[totals == 0] = 1
                chunk *= 10000.0 / totals
                np.log1p(chunk, out=chunk)
            yield start, end, chunk
            start = end

    print('Computing incremental PCA: %d components, %d cells per chunk (%s).'
          % (sample_pca_components, sample_pca_chunk_size, sample_pca_input))
    pca = IncrementalPCA(n_components=sample_pca_components)
    for _, _, chunk in batches():
        pca.partial_fit(chunk)
    del chunk

    output_dir = Path(output_dir)
    embedding_path = output_dir / 'pca.npy'
    embedding = np.lib.format.open_memmap(
        embedding_path, mode='w+', dtype=np.float32,
        shape=(data.n_obs, sample_pca_components))
    # Transform only after fitting every batch, using the same final basis.
    for start, end, chunk in batches():
        embedding[start:end] = pca.transform(chunk)
        if not np.isfinite(embedding[start:end]).all():
            raise ValueError('PCA coordinates contain NaN or Inf after float32 conversion')
    embedding.flush()
    del embedding
    np.savez(output_dir / 'pca_model.npz', components=pca.components_,
             mean=pca.mean_, explained_variance=pca.explained_variance_,
             explained_variance_ratio=pca.explained_variance_ratio_)
    return np.load(embedding_path, mmap_mode='r'), {
        'source': 'computed', 'mode': sample_pca_mode,
        'components': sample_pca_components, 'chunk_size': sample_pca_chunk_size,
        'input': sample_pca_input,
        'preprocessing': 'log1p(CP10K)' if sample_pca_input == 'counts' else 'none',
        'dtype': 'float32', 'working_dtype': 'float64', 'gene_selection': 'all',
        'embedding': 'pca.npy', 'model': 'pca_model.npz',
    }


def _sample_random(n_cells, n_sample, seed):
    """Select cell indices uniformly without replacement."""
    return np.random.default_rng(seed).choice(n_cells, n_sample, replace=False)


def _sample_scsampler(embedding, n_sample, seed):
    """Select diverse cell indices without changing the caller's NumPy RNG."""
    from scsampler import scsampler

    # scsampler seeds the global RNG; preserve the caller's state.
    state = np.random.get_state()
    try:
        return scsampler(embedding, n_obs=n_sample, random_state=seed, copy=False)
    finally:
        np.random.set_state(state)


def sample(counts_fn, output_dir, sample_method='random', sample_times=1,
           sample_fraction=None, sample_seed=42, sample_prefix=None,
           sample_pca_mode='auto', sample_pca_components=30,
           sample_pca_chunk_size=2000, sample_pca_input='counts',
           sample_numgenes=2000, sample_hvg_mode='auto',
           sample_hvg_chunk_size=2000, sample_export_chunk_size=2000):
    """Write independent H5AD draws and return a manifest path.

    scsampler reuses X_pca or computes a shared incremental PCA embedding.
    PCA preprocessing affects temporary batches only, never the sampled counts.
    Count matrices are exported in chunks, not materialized as a whole sample.
    Dense/CSR input is written as CSR; CSC stays CSC for bounded column reads.
    Filenames are <sample_prefix>_<draw:03d>.h5ad; the default prefix is
    <input-stem>_sample. The prefix must not include a directory.
    A shared <sample_prefix>.overdispersed_genes.txt is reused or calculated
    once from the full input, not separately per draw. Pass it to prepare's
    genes_file to skip HVG selection there. sample_numgenes=0 skips this step.
    """
    if sample_method not in ('random', 'scsampler'):
        raise ValueError('sample_method must be random or scsampler')
    if not isinstance(sample_times, int) or sample_times < 1:
        raise ValueError('sample_times must be a positive integer')
    if sample_fraction is None or not 0 < sample_fraction <= 1:
        raise ValueError('sample_fraction must be in (0, 1]')
    if not isinstance(sample_export_chunk_size, int) or sample_export_chunk_size < 1:
        raise ValueError('sample_export_chunk_size must be a positive integer')
    sample_prefix, filenames = _sample_filenames(counts_fn, sample_times, sample_prefix)

    seeds = np.random.SeedSequence(sample_seed).generate_state(sample_times)
    data = ad.read_h5ad(counts_fn, backed='r')
    try:
        n_sample = int(data.n_obs * sample_fraction)
        if n_sample < 1:
            raise ValueError('sample_fraction selects fewer than one cell')
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        hvg_info = _sample_hvgs(data, output_dir, sample_prefix, sample_numgenes,
                                sample_hvg_mode, sample_hvg_chunk_size)
        embedding, pca_info = None, None
        if sample_method == 'scsampler' and n_sample < data.n_obs:
            embedding, pca_info = _sample_embedding(
                data, output_dir, sample_pca_mode, sample_pca_components,
                sample_pca_chunk_size, sample_pca_input)
        manifest = {
            'input': str(Path(counts_fn).resolve()),
            'sample_method': sample_method, 'sample_fraction': sample_fraction,
            'sample_times': sample_times, 'sample_seed': sample_seed,
            'sample_prefix': sample_prefix,
            'pca': pca_info, 'hvg': hvg_info,
            'export': {'chunk_size': sample_export_chunk_size, 'sparse': True},
            'n_cells': data.n_obs, 'n_genes': data.n_vars, 'samples': [],
        }
        for files, seed in zip(filenames, seeds):
            seed = int(seed)
            if n_sample == data.n_obs:
                indices = np.arange(data.n_obs)
            elif sample_method == 'random':
                indices = _sample_random(data.n_obs, n_sample, seed)
            else:
                indices = _sample_scsampler(embedding, n_sample, seed)
            indices = np.sort(np.asarray(indices, dtype=np.int64))
            if len(indices) != n_sample or len(np.unique(indices)) != n_sample:
                raise ValueError('Sampler did not return the requested unique cells')
            _sample_export(data, indices, output_dir / files['counts'],
                           sample_export_chunk_size)
            np.save(output_dir / files['cell_indices'], indices)
            manifest['samples'].append({
                **files, 'sample_seed': seed, 'n_cells': n_sample,
            })
        manifest_path = output_dir / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        return str(manifest_path)
    finally:
        data.file.close()

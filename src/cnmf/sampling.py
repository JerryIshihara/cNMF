"""Cell sampling that preserves input genes, counts, and annotations."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings
import yaml

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


SAMPLE_MEMORY_BUFFER_MIN = 4 * 1024**3
SAMPLE_MEMORY_BUFFER_FRACTION = 0.10
SAMPLE_DENSE_EXPORT_MAX_ROWS = 2000
# Internal PCA settings; sampling exposes only the embedding key.
SAMPLE_PCA_COMPONENTS = 30
SAMPLE_PCA_CHUNK_SIZE = 2000


def _sample_memory_budget():
    """Reserve 4 GiB or 10% of available RAM, capped at half on small hosts."""
    try:
        info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(info['MemAvailable'].split()[0]) * 1024
    except (OSError, KeyError, ValueError):
        try:
            available = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        except (OSError, ValueError, AttributeError):
            available = 1024**3

    # Respect Linux cgroup limits as well as host RAM, including parent groups.
    try:
        groups = Path('/proc/self/cgroup').read_text().splitlines()
    except OSError:
        groups = []
    for group in groups:
        _, controllers, relative = group.split(':', 2)
        if controllers == '':
            root, files = Path('/sys/fs/cgroup'), ('memory.max', 'memory.current')
        elif 'memory' in controllers.split(','):
            root = Path('/sys/fs/cgroup/memory')
            files = ('memory.limit_in_bytes', 'memory.usage_in_bytes')
        else:
            continue
        directory = root / relative.lstrip('/')
        while directory.is_relative_to(root):
            try:
                limit = int((directory / files[0]).read_text())
                used = int((directory / files[1]).read_text())
                available = min(available, max(0, limit - used))
            except (OSError, ValueError):
                pass  # An absent limit or cgroup v2's "max" is unrestricted.
            directory = directory.parent
    available = max(0, available)
    reserve = min(available // 2,
                  max(SAMPLE_MEMORY_BUFFER_MIN, int(available * SAMPLE_MEMORY_BUFFER_FRACTION)))
    return available - reserve


def _sample_chunk_size(shape, dtype, memory_budget, axis=0):
    """Budget dense values, sparse indices and temporary copies per read."""
    bytes_per_unit = shape[1-axis] * 4 * (max(np.dtype(dtype).itemsize, 8) + 8)
    size = memory_budget // max(1, bytes_per_unit)
    if size < 1:
        raise ValueError('Sampling memory budget cannot fit one matrix %s' %
                         ('row' if axis == 0 else 'column'))
    return min(shape[axis], size)


def _sample_combine_runs(destination, source_runs, components=None):
    """Stage sampled replicates under unique IDs for ordinary combine.

    Rebuild links and metadata on every call. Spectra reading, missing-file
    handling and merging belong to combine_nmf, not this helper.
    """
    from .cnmf import cNMF, load_df_from_npz

    sources = [Path(p).resolve() for p in source_runs]
    target = (Path(destination.output_dir) / destination.name).resolve()
    receipt_path = target / 'cnmf_tmp' / 'pooling.json'
    if not sources or len(set(sources)) != len(sources) or target in sources:
        raise ValueError('Source runs must be distinct and different from destination')
    if not receipt_path.exists() and any(p.is_file() for p in target.rglob('*')):
        raise FileExistsError('Destination must be empty or an existing pooled run')

    manifests, settings = [], None
    for root in sources:
        prefix = root / 'cnmf_tmp' / root.name
        params = load_df_from_npz(str(prefix) + '.nmf_params.df.npz')
        with open(str(prefix) + '.nmf_idvrun_params.yaml') as stream:
            current = yaml.safe_load(stream)
        if settings is not None and current != settings:
            raise ValueError('Source NMF settings differ: %s' % root)
        settings = current
        if params.duplicated(['n_components', 'iter']).any():
            raise ValueError('Duplicate replicate IDs: %s' % root)
        manifests.append((root, prefix, params))
    if type(components) is int:
        ks = [components]
    elif components is None:
        ks = sorted(set(manifests[0][2].n_components))
        if any(set(p.n_components) != set(ks) for _, _, p in manifests):
            raise ValueError('Source K sets differ; specify components explicitly')
    else:
        ks = sorted(set(components))
    if not ks:
        raise ValueError('No components selected')

    records, rows = [], []
    with TemporaryDirectory(prefix='.combine-', dir=target.parent) as folder:
        staged = cNMF(folder, destination.name)
        for k in ks:
            global_iter = 0
            for draw, (root, prefix, params) in enumerate(manifests):
                found = 0
                for _, row in params[params.n_components == k].sort_values('iter').iterrows():
                    local_iter = int(row['iter'])
                    source = Path(str(prefix) + '.spectra.k_%d.iter_%d.df.npz' % (k, local_iter))
                    Path(staged.paths['iter_spectra'] % (k, global_iter)).symlink_to(source)
                    item = row.to_dict()
                    item['iter'] = global_iter
                    rows.append(item)
                    records.append(dict(k=int(k), global_iter=global_iter, draw=draw,
                                        source_run=str(root), local_iter=local_iter,
                                        nmf_seed=int(row['nmf_seed'])))
                    global_iter += 1
                    found += 1
                if not found:
                    raise ValueError('No spectra for K=%s in %s' % (k, root))
        staged.save_nmf_iter_params(pd.DataFrame(rows), settings)
        stage = Path(folder) / destination.name / 'cnmf_tmp'
        pd.DataFrame(records).to_csv(stage / 'pooled_replicates.tsv', sep='\t', index=False)

        # Delete only files from the previous mapping and known downstream outputs.
        if receipt_path.exists():
            previous = json.loads(receipt_path.read_text())
            for record in previous['records']:
                link = Path(destination.paths['iter_spectra'] % (record['k'], record['global_iter']))
                if link.is_symlink():
                    link.unlink()
                elif link.exists():
                    raise FileExistsError('Refusing to remove non-link replicate: %s' % link)
            for key, template in destination.paths.items():
                if key in ('iter_spectra', 'iter_usages') or '%' not in template:
                    continue
                pattern = Path(template.replace('%d', '*').replace('%s', '*'))
                for path in pattern.parent.glob(pattern.name):
                    path.unlink()
            for path in (target / 'cnmf_tmp').glob(destination.name + '.spectra.k_*.complete.json'):
                path.unlink()
            for path in target.glob(destination.name + '.k_*_dt_*.*'):
                path.unlink()
            for key in ('k_selection_plot', 'k_selection_stats'):
                Path(destination.paths[key]).unlink(missing_ok=True)
        for path in stage.iterdir():
            path.replace(target / 'cnmf_tmp' / path.name)
        receipt_path.write_text(json.dumps(dict(records=records,
            scaling_validation='not_verified', expression_inputs='not_assigned'), indent=2))
    warnings.warn('Spectra pooled; expression inputs are not assigned. '
                  'Full-data consensus/refit integration must be configured separately.')


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


def _write_sample_matrix(group, name, matrix, indices, chunk_size, pairwise=False,
                         memory_budget=None):
    """Stream a selected matrix in its sparse storage direction.

    Dense input becomes CSR without rounding or thresholding. Keep CSC input
    column-compressed: row indexing a backed CSC matrix can read the entire
    matrix. Its column blocks use the equivalent dense-element chunk budget.
    Auto mode sizes each matrix independently, including wider raw/graphs.
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
    auto_chunk = chunk_size is None
    if auto_chunk:
        if memory_budget is None:
            memory_budget = _sample_memory_budget()
        chunk_size = _sample_chunk_size(matrix.shape, matrix.dtype, memory_budget,
                                        axis=1 if fmt == 'csc' else 0)
        # Long h5py fancy-index selections can be slower despite fewer reads.
        if isinstance(matrix, h5py.Dataset):
            chunk_size = min(chunk_size, SAMPLE_DENSE_EXPORT_MAX_ROWS)
        chunk_size = max(1, min(chunk_size, size))
    elif fmt == 'csc':
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
    info = {'axis': 'columns' if fmt == 'csc' else 'rows', 'size': chunk_size}
    if auto_chunk:
        info['memory_budget_bytes'] = memory_budget
    return info


def _sample_export(data, indices, filename, chunk_size, memory_budget=None):
    """Write one ordinary sparse H5AD without materializing the sampled X.

    X, layers, raw/X and pairwise cell matrices are streamed separately.
    Annotations already loaded by AnnData's backed reader stay in memory;
    this bounds export matrix buffers, not the entire sampling/prepare RAM.
    Auto mode refreshes available memory before each matrix, after metadata
    and sampling/PCA allocations, unless a low-level budget is supplied.
    Publish the file only after every matrix has been written successfully.
    """
    filename = Path(filename)
    chunks = {}
    selected = data[indices]
    metadata = ad.AnnData(obs=selected.obs.copy(), var=selected.var.copy(),
                         obsm=dict(selected.obsm), varm=dict(selected.varm),
                         varp=dict(selected.varp), uns=dict(selected.uns))
    with TemporaryDirectory(prefix='.' + filename.stem + '-', dir=filename.parent) as folder:
        temporary = Path(folder) / filename.name
        metadata.write_h5ad(temporary)
        with h5py.File(temporary, 'a') as output:
            chunks['X'] = _write_sample_matrix(output, 'X', data.X, indices, chunk_size,
                                               memory_budget=memory_budget)
            for name, matrix in data.layers.items():
                if name is None:  # Newer AnnData exposes X here; it was written above.
                    continue
                chunks['layers/' + name] = _write_sample_matrix(
                    output['layers'], name, matrix, indices, chunk_size,
                    memory_budget=memory_budget)
            for name, matrix in data.obsp.items():
                chunks['obsp/' + name] = _write_sample_matrix(
                    output['obsp'], name, matrix, indices, chunk_size,
                    pairwise=True, memory_budget=memory_budget)
            if data.raw is not None:
                raw = output.create_group('raw')
                # Gene annotations are unchanged: raw can contain more genes
                # than X, and these must not be intersected or reordered.
                with h5py.File(data.filename, 'r') as source:
                    raw.attrs.update(source['raw'].attrs)
                    for name in ('var', 'varm'):
                        if name in source['raw']:
                            source.copy(source['raw'][name], raw, name=name)
                chunks['raw/X'] = _write_sample_matrix(
                    raw, 'X', data.raw.X, indices, chunk_size, memory_budget=memory_budget)
        temporary.replace(filename)
    return chunks


def _sample_hvgs(data, output_dir, sample_prefix, sample_numgenes=2000,
                 sample_hvg_mode='auto', sample_hvg_chunk_size=None, memory_budget=None):
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
    if sample_hvg_chunk_size is not None and (not isinstance(sample_hvg_chunk_size, int)
                                              or sample_hvg_chunk_size < 1):
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

        if sample_hvg_chunk_size is None:
            if memory_budget is None:
                memory_budget = _sample_memory_budget()
            sample_hvg_chunk_size = _sample_chunk_size(data.shape, data.X.dtype, memory_budget)
            info['memory_budget_bytes'] = memory_budget
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


def _sample_embedding(data, output_dir, sample_embedding_key='X_pca'):
    """Reuse an obsm embedding or compute missing X_pca from counts.

    Newly computed coordinates are stored in a float32 NPY memmap, not added
    to the input H5AD. Fitting uses float64 batches in input cell order and all
    input genes, with CP10K/log1p on temporary counts batches only. Other
    embedding keys must already exist; their preprocessing is left unchanged.
    """
    chunk_size = SAMPLE_PCA_CHUNK_SIZE
    if sample_embedding_key in data.obsm:
        embedding = np.asarray(data.obsm[sample_embedding_key])
        if (embedding.ndim != 2 or embedding.shape[0] != data.n_obs
                or embedding.shape[1] < 1 or embedding.dtype.kind not in 'fiu'):
            raise ValueError('obsm["%s"] must be a real cells-by-components matrix' % sample_embedding_key)
        for start in range(0, data.n_obs, chunk_size):
            if not np.isfinite(embedding[start:start + chunk_size]).all():
                raise ValueError('obsm["%s"] contains NaN or Inf' % sample_embedding_key)
        print('Reusing obsm["%s"] with %d components.' % (sample_embedding_key, embedding.shape[1]))
        return embedding, {
            'source': 'obsm/' + sample_embedding_key, 'mode': 'auto',
            'components': embedding.shape[1], 'dtype': str(embedding.dtype),
            # An existing embedding does not establish its preprocessing history.
            'preprocessing': None,
        }

    if sample_embedding_key != 'X_pca':
        raise ValueError('Requested embedding obsm["%s"] is missing; only X_pca can be computed from counts'
                         % sample_embedding_key)

    from sklearn.decomposition import IncrementalPCA

    n_components = min(SAMPLE_PCA_COMPONENTS, data.n_obs - 1, data.n_vars)
    if n_components < 1:
        raise ValueError('Computing PCA requires at least two cells and one gene')

    def batches():
        start = 0
        while start < data.n_obs:
            end = min(start + chunk_size, data.n_obs)
            # Merge a short final batch so every fit has at least n_components rows.
            if data.n_obs - end < n_components:
                end = data.n_obs
            chunk = data.X[start:end]
            if sparse.issparse(chunk):
                chunk = chunk.astype(np.float64).toarray()
            else:
                chunk = np.array(chunk, dtype=np.float64, copy=True)
            if not np.isfinite(chunk).all():
                raise ValueError('PCA input contains NaN or Inf')
            if (chunk < 0).any():
                raise ValueError('Computing PCA from counts requires nonnegative values')
            totals = chunk.sum(axis=1, keepdims=True)
            totals[totals == 0] = 1
            chunk *= 10000.0 / totals
            np.log1p(chunk, out=chunk)
            yield start, end, chunk
            start = end

    print('Computing incremental PCA: %d components, %d cells per chunk (counts).'
          % (n_components, chunk_size))
    pca = IncrementalPCA(n_components=n_components)
    for _, _, chunk in batches():
        pca.partial_fit(chunk)
    del chunk

    output_dir = Path(output_dir)
    embedding_path = output_dir / 'pca.npy'
    embedding = np.lib.format.open_memmap(
        embedding_path, mode='w+', dtype=np.float32,
        shape=(data.n_obs, n_components))
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
        'source': 'computed', 'mode': 'auto',
        'components': n_components, 'chunk_size': chunk_size,
        'input': 'counts', 'preprocessing': 'log1p(CP10K)',
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
           sample_embedding_key='X_pca',
           sample_numgenes=2000, sample_hvg_mode='auto',
           sample_hvg_chunk_size=None, sample_export_chunk_size=None):
    """Write independent H5AD draws and return a manifest path.

    scsampler reuses obsm[sample_embedding_key]. Missing X_pca is computed
    from counts using incremental PCA; other keys must exist. Random sampling
    ignores the embedding key.
    PCA preprocessing affects temporary batches only, never the sampled counts.
    Count matrices are exported in chunks, not materialized as a whole sample.
    HVG/export chunks default to automatic buffer sizing; positive integers
    override it. PCA retains its existing fixed batch boundaries.
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
    if sample_export_chunk_size is not None and (not isinstance(sample_export_chunk_size, int)
                                                 or sample_export_chunk_size < 1):
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
                data, output_dir, sample_embedding_key)
        manifest = {
            'input': str(Path(counts_fn).resolve()),
            'sample_method': sample_method, 'sample_fraction': sample_fraction,
            'sample_times': sample_times, 'sample_seed': sample_seed,
            'sample_prefix': sample_prefix,
            'pca': pca_info, 'hvg': hvg_info,
            'export': {'chunk_size': sample_export_chunk_size, 'sparse': True},
            'n_cells': data.n_obs, 'n_genes': data.n_vars, 'samples': [],
        }
        if sample_export_chunk_size is None:
            manifest['export'].update(memory_buffer_min_bytes=SAMPLE_MEMORY_BUFFER_MIN,
                                      memory_buffer_fraction=SAMPLE_MEMORY_BUFFER_FRACTION)
            print('Auto-sizing export chunks from available RAM minus a memory reserve.')
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
            export_chunks = _sample_export(data, indices, output_dir / files['counts'],
                                            sample_export_chunk_size)
            np.save(output_dir / files['cell_indices'], indices)
            manifest['samples'].append({
                **files, 'sample_seed': seed, 'n_cells': n_sample,
            })
            if sample_export_chunk_size is None:
                manifest['samples'][-1]['export_chunks'] = export_chunks
        manifest_path = output_dir / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        return str(manifest_path)
    finally:
        data.file.close()

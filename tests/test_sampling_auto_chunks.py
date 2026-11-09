"""Auto chunking must change I/O granularity, not selected cells or counts.

Fixtures are generated here from fixed NumPy seeds. Small mocked RAM budgets
exercise the same multi-chunk paths as large inputs without allocating them.
"""

import inspect
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import cnmf.sampling as sampling
from cnmf.cnmf import cNMF, compute_tpm, get_highvar_genes_sparse, main


def _dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


@pytest.mark.parametrize('available_gib,groups,limits,expected_gib', [
    (16, '', {}, 12),
    (60, '', {}, 54),
    (4, '', {}, 2),
    (0, '', {}, 0),
    (256, '0::/job/worker', {
        '/sys/fs/cgroup/job/worker/memory.max': 'max',
        '/sys/fs/cgroup/job/memory.max': str(12 * 1024**3),
        '/sys/fs/cgroup/job/memory.current': str(4 * 1024**3),
    }, 4),
    (256, '5:memory:/job/worker', {
        '/sys/fs/cgroup/memory/job/worker/memory.limit_in_bytes': str(10 * 1024**3),
        '/sys/fs/cgroup/memory/job/worker/memory.usage_in_bytes': str(1024**3),
        '/sys/fs/cgroup/memory/job/memory.limit_in_bytes': str(6 * 1024**3),
        '/sys/fs/cgroup/memory/job/memory.usage_in_bytes': str(2 * 1024**3),
    }, 2),
])
def test_memory_budget_reserves_headroom_and_respects_parent_cgroups(
        monkeypatch, available_gib, groups, limits, expected_gib):
    files = {'/proc/meminfo': f'MemAvailable: {available_gib * 1024**2} kB\n',
             '/proc/self/cgroup': groups, **limits}

    def read_text(path, *args, **kwargs):
        if str(path) not in files:
            raise FileNotFoundError(str(path))
        return files[str(path)]

    monkeypatch.setattr(Path, 'read_text', read_text)
    assert sampling._sample_memory_budget() == expected_gib * 1024**3


@pytest.mark.parametrize('sysconf_available,expected', [(True, 1024**3), (False, 512 * 1024**2)])
def test_memory_budget_falls_back_without_linux_memory_files(monkeypatch, sysconf_available, expected):
    def missing(*args, **kwargs):
        raise FileNotFoundError

    def sysconf(name):
        if not sysconf_available:
            raise OSError
        return {'SC_AVPHYS_PAGES': 524288, 'SC_PAGE_SIZE': 4096}[name]

    monkeypatch.setattr(Path, 'read_text', missing)
    monkeypatch.setattr(sampling.os, 'sysconf', sysconf)
    assert sampling._sample_memory_budget() == expected


@pytest.mark.parametrize('shape,dtype,budget,axis,expected', [
    ((100, 20), 'float32', 20 * 64 * 7, 0, 7),
    ((100, 20), 'float64', 20 * 64 * 7, 0, 7),
    ((100, 20), 'uint16', 20 * 64 * 7, 0, 7),
    ((100, 20), 'float32', 100 * 64 * 3, 1, 3),
    ((100, 20), 'float32', 2**40, 0, 100),
    ((1000000, 20), 'float32', 2**40, 0, 1000000),
])
def test_auto_chunk_budget_accounts_for_workspace_and_axis(
        shape, dtype, budget, axis, expected):
    assert sampling._sample_chunk_size(shape, np.dtype(dtype), budget, axis) == expected


def test_auto_chunk_rejects_budget_smaller_than_one_row():
    with pytest.raises(ValueError):
        sampling._sample_chunk_size((100, 20), np.dtype('float32'), 20 * 64 - 1)


@pytest.mark.parametrize('matrix_type', ['dense', 'csr', 'csc'])
@pytest.mark.parametrize('pairwise', [False, True])
def test_auto_export_reads_only_budgeted_rows_or_columns(tmp_path, matrix_type, pairwise):
    shape = (53, 53 if pairwise else 41)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    matrix = values if matrix_type == 'dense' else getattr(sparse, matrix_type + '_matrix')(values)
    indices = np.arange(1, shape[0], 2)
    axis = 1 if matrix_type == 'csc' else 0
    budget = shape[1 - axis] * 64 * 3
    reads = []

    class GuardedMatrix:
        format = matrix_type
        shape = matrix.shape
        dtype = matrix.dtype

        def __array__(self, *args, **kwargs):
            pytest.fail('Auto chunking must not materialize the whole matrix')

        def __getitem__(self, key):
            selected = key[axis]
            count = selected.stop - selected.start if isinstance(selected, slice) else len(selected)
            assert count <= 3
            reads.append(count)
            return matrix[key]

    width = len(indices) if pairwise else shape[1]
    path = tmp_path / 'bounded.h5ad'
    ad.AnnData(obs=pd.DataFrame(index=indices.astype(str)),
               var=pd.DataFrame(index=np.arange(width).astype(str))).write_h5ad(path)
    with h5py.File(path, 'a') as handle:
        receipt = sampling._write_sample_matrix(
            handle, 'X', GuardedMatrix(), indices, None,
            pairwise=pairwise, memory_budget=budget)
    assert receipt == {'axis': 'columns' if axis else 'rows', 'size': 3,
                       'memory_budget_bytes': budget}
    assert len(reads) > 1
    expected = values[indices][:, indices] if pairwise else values[indices]
    np.testing.assert_array_equal(_dense(ad.read_h5ad(path).X), expected)


def test_dense_backed_export_caps_auto_but_honors_larger_manual_chunks(tmp_path):
    # A real dense HDF5 dataset exercises its fancy-index path. Using only
    # three genes keeps the fixture small while spanning multiple row chunks.
    values = np.arange(10002 * 3, dtype=np.float32).reshape(10002, 3)
    path = tmp_path / 'dense.h5ad'
    ad.AnnData(values).write_h5ad(path)
    indices = np.arange(1, len(values), 2)
    data = ad.read_h5ad(path, backed='r')
    try:
        assert isinstance(data.X, h5py.Dataset)
        for chunk_size, expected_size in ((None, 2000), (4096, 4096)):
            output = tmp_path / f'export_{chunk_size}.h5ad'
            receipt = sampling._sample_export(
                data, indices, output, chunk_size, memory_budget=16 * 1024**2)
            expected_receipt = {'axis': 'rows', 'size': expected_size}
            if chunk_size is None:
                expected_receipt['memory_budget_bytes'] = 16 * 1024**2
            assert receipt['X'] == expected_receipt
            actual = ad.read_h5ad(output)
            np.testing.assert_array_equal(_dense(actual.X), values[indices])
            assert actual.obs_names.tolist() == data.obs_names[indices].tolist()
    finally:
        data.file.close()


@pytest.mark.parametrize('matrix_type,dtype', [
    ('dense', np.float32), ('csr', np.uint16), ('csc', np.float64),
])
def test_auto_export_preserves_draws_annotations_and_all_matrix_slots(
        tmp_path, monkeypatch, matrix_type, dtype):
    rng = np.random.default_rng(481)

    def convert(values):
        values = values.astype(dtype)
        return values if matrix_type == 'dense' else getattr(sparse, matrix_type + '_matrix')(values)

    values = rng.poisson(0.7, (23, 9)).astype(dtype)
    source = ad.AnnData(convert(values),
                       obs=pd.DataFrame({'donor': pd.Categorical(['a', 'b'] * 11 + ['a'])},
                                        index=[f'cell_{i}' for i in range(23)]),
                       var=pd.DataFrame(index=[f'gene_{i}' for i in range(9)]))
    source.layers['counts'] = sparse.csr_matrix(values)
    source.layers['dense'] = values.astype(np.float64) + 0.125
    source.obsm['X_pca'] = rng.normal(size=(23, 3)).astype(np.float32)
    source.obsp['graph'] = sparse.csr_matrix(rng.integers(0, 3, (23, 23)))
    source.obsp['csc_graph'] = source.obsp['graph'].tocsc()
    source.obsp['dense_graph'] = source.obsp['graph'].toarray()
    source.uns['provenance'] = {'seed': 481}
    source.raw = ad.AnnData(convert(rng.poisson(0.5, (23, 41))),
                            obs=source.obs.copy(),
                            var=pd.DataFrame(index=[f'raw_{i}' for i in range(41)]))
    path = tmp_path / 'counts.h5ad'
    source.write_h5ad(path)
    # Raw expression has more genes than X; sizing only from X would exceed
    # this budget. Pairwise matrices need their own width as well.
    budget = 41 * 64 * 2
    budget_calls = []

    def memory_budget():
        budget_calls.append(True)
        return budget

    monkeypatch.setattr(sampling, '_sample_memory_budget', memory_budget)
    auto_path = sampling.sample(path, tmp_path / 'auto', sample_fraction=0.7,
                                sample_times=3, sample_numgenes=0)
    auto = json.loads(Path(auto_path).read_text())
    expected_budget_calls = sum(len(draw['export_chunks']) for draw in auto['samples'])
    assert len(budget_calls) == expected_budget_calls
    manual_path = sampling.sample(path, tmp_path / 'manual', sample_fraction=0.7,
                                  sample_times=3, sample_numgenes=0,
                                  sample_hvg_chunk_size=7, sample_export_chunk_size=7)
    auto, manual = (json.loads(Path(p).read_text()) for p in (auto_path, manual_path))
    assert len(budget_calls) == expected_budget_calls  # Explicit chunks do not query RAM.
    assert auto['export']['chunk_size'] is None
    assert 'memory_budget_bytes' not in auto['export']
    assert auto['export']['memory_buffer_min_bytes'] == 4 * 1024**3
    assert auto['export']['memory_buffer_fraction'] == 0.1
    assert manual['export'] == {'chunk_size': 7, 'sparse': True}

    for auto_draw, manual_draw in zip(auto['samples'], manual['samples']):
        indices = np.load(Path(auto_path).parent / auto_draw['cell_indices'])
        np.testing.assert_array_equal(indices, np.load(Path(manual_path).parent / manual_draw['cell_indices']))
        expected = source[indices].copy()
        actual = ad.read_h5ad(Path(auto_path).parent / auto_draw['counts'])
        manual_data = ad.read_h5ad(Path(manual_path).parent / manual_draw['counts'])
        pd.testing.assert_frame_equal(actual.obs, expected.obs)
        pd.testing.assert_frame_equal(actual.var, expected.var)
        pd.testing.assert_frame_equal(actual.raw.var, expected.raw.var)
        np.testing.assert_array_equal(actual.obsm['X_pca'], expected.obsm['X_pca'])
        assert actual.uns == expected.uns
        matrices = {'X': (actual.X, expected.X, manual_data.X),
                    'raw/X': (actual.raw.X, expected.raw.X, manual_data.raw.X)}
        source_matrices = {'X': source.X, 'raw/X': source.raw.X}
        for slot in ('layers', 'obsp'):
            for name, original in getattr(source, slot).items():
                if name is None:  # Newer AnnData also exposes X in layers.
                    continue
                key = f'{slot}/{name}'
                matrices[key] = tuple(getattr(data, slot)[name]
                                      for data in (actual, expected, manual_data))
                source_matrices[key] = original
        assert set(auto_draw['export_chunks']) == set(matrices)
        for key, (result, expected_matrix, manual_matrix) in matrices.items():
            assert result.dtype == expected_matrix.dtype
            np.testing.assert_array_equal(_dense(result), _dense(expected_matrix))
            np.testing.assert_array_equal(_dense(result), _dense(manual_matrix))
            original = source_matrices[key]
            axis = 1 if getattr(original, 'format', None) == 'csc' else 0
            receipt = auto_draw['export_chunks'][key]
            assert receipt['axis'] == ('columns' if axis else 'rows')
            assert receipt['memory_budget_bytes'] == budget
            assert receipt['size'] * original.shape[1 - axis] * 64 <= budget


def test_auto_hvgs_match_explicit_chunks_and_original_cnmf(tmp_path, monkeypatch):
    rng = np.random.default_rng(809)
    values = rng.negative_binomial(3, rng.uniform(0.15, 0.85, 67), (143, 67))
    source = ad.AnnData(sparse.csr_matrix(values))
    source.var_names = [f'gene_{i}' for i in range(67)]
    path = tmp_path / 'hvg_counts.h5ad'
    source.write_h5ad(path)
    expected, _ = get_highvar_genes_sparse(compute_tpm(source).X, numgenes=13)
    expected_genes = source.var_names[expected.high_var].tolist()
    monkeypatch.setattr(sampling, '_sample_memory_budget', lambda: 67 * 64 * 17)
    for name, overrides in [('auto', {}), ('manual', {'sample_hvg_chunk_size': 61})]:
        manifest_path = sampling.sample(path, tmp_path / name, sample_fraction=0.5,
                                        sample_numgenes=13, **overrides)
        manifest = json.loads(Path(manifest_path).read_text())
        assert manifest['hvg']['chunk_size'] == (17 if name == 'auto' else 61)
        assert (Path(manifest_path).parent / manifest['hvg']['genes_file']).read_text().splitlines() == expected_genes


def test_auto_budget_is_refreshed_after_embedding_and_for_each_draw(tmp_path, monkeypatch):
    values = np.random.default_rng(309).poisson(3, (48, 9)).astype(np.float32)
    path = tmp_path / 'counts.h5ad'
    ad.AnnData(sparse.csr_matrix(values)).write_h5ad(path)
    bytes_per_row = 9 * 64
    state = {'budget': bytes_per_row * 12, 'draw': 0}
    events = []

    def memory_budget():
        events.append(('budget', state['budget']))
        return state['budget']

    def embedding(data, *args, **kwargs):
        events.append(('pca', None))
        # Emulate memory retained by PCA/the embedding before exporting.
        state['budget'] = bytes_per_row * 5
        return np.zeros((data.n_obs, 2), dtype=np.float32), {'source': 'computed'}

    def draw(embedding, n_sample, seed):
        state['draw'] += 1
        events.append(('draw', state['draw']))
        if state['draw'] == 2:
            state['budget'] = bytes_per_row * 2
        return np.arange(n_sample)

    monkeypatch.setattr(sampling, '_sample_memory_budget', memory_budget)
    monkeypatch.setattr(sampling, '_sample_embedding', embedding)
    monkeypatch.setattr(sampling, '_sample_scsampler', draw)
    manifest_path = sampling.sample(path, tmp_path / 'out', 'scsampler',
                                    sample_times=2, sample_fraction=1 / 3,
                                    sample_numgenes=4)
    manifest = json.loads(Path(manifest_path).read_text())
    assert events == [('budget', bytes_per_row * 12), ('pca', None),
                      ('draw', 1), ('budget', bytes_per_row * 5),
                      ('draw', 2), ('budget', bytes_per_row * 2)]
    assert manifest['hvg']['chunk_size'] == 12
    for draw_info, rows in zip(manifest['samples'], (5, 2)):
        assert draw_info['export_chunks']['X'] == {
            'axis': 'rows', 'size': rows, 'memory_budget_bytes': bytes_per_row * rows}
        actual = ad.read_h5ad(Path(manifest_path).parent / draw_info['counts'])
        np.testing.assert_array_equal(_dense(actual.X), values[:16])


@pytest.mark.parametrize('option', ['sample_hvg_chunk_size', 'sample_export_chunk_size'])
def test_zero_is_not_an_auto_chunk_sentinel(tmp_path, option):
    path = tmp_path / 'counts.h5ad'
    ad.AnnData(np.ones((10, 4), dtype=np.float32)).write_h5ad(path)
    with pytest.raises(ValueError, match=option):
        sampling.sample(path, tmp_path / 'out', sample_fraction=0.5, **{option: 0})


def test_public_sampling_defaults_keep_pca_settings_internal():
    for function in (sampling.sample, cNMF.sample):
        parameters = inspect.signature(function).parameters
        assert parameters['sample_hvg_chunk_size'].default is None
        assert parameters['sample_export_chunk_size'].default is None
        assert parameters['sample_embedding_key'].default == 'X_pca'
        assert not any(name.startswith('sample_pca_') for name in parameters)
    assert sampling.SAMPLE_PCA_COMPONENTS == 30
    assert sampling.SAMPLE_PCA_CHUNK_SIZE == 2000


def test_cli_uses_auto_export_by_default(tmp_path, monkeypatch):
    path = tmp_path / 'counts.h5ad'
    ad.AnnData(np.arange(60, dtype=np.float32).reshape(15, 4)).write_h5ad(path)
    monkeypatch.setattr(sampling, '_sample_memory_budget', lambda: 4 * 64 * 3)
    monkeypatch.setattr('sys.argv', [
        'cnmf', 'sample', '--counts', str(path), '--output-dir', str(tmp_path),
        '--name', 'cli', '--sample-fraction', '0.5', '--sample-numgenes', '0',
    ])
    main()
    manifest = json.loads((tmp_path / 'cli' / 'sampling' / 'manifest.json').read_text())
    assert manifest['export']['chunk_size'] is None
    assert manifest['samples'][0]['export_chunks']['X'] == {
        'axis': 'rows', 'size': 3, 'memory_budget_bytes': 4 * 64 * 3}

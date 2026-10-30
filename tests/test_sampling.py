"""Small deterministic synthetic data; no downloaded fixtures required."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.decomposition import IncrementalPCA

import cnmf.sampling as sampling
import cnmf.cnmf as cnmf_module

from cnmf.cnmf import (
    cNMF, compute_tpm, get_highvar_genes, get_highvar_genes_sparse, main,
)
from cnmf.sampling import _sample_embedding, _sample_hvgs, sample


@pytest.fixture
def counts(tmp_path):
    rng = np.random.default_rng(123)
    data = ad.AnnData(sparse.csr_matrix(rng.poisson(3, (60, 8))))
    data.obsm['X_pca'] = rng.normal(size=(60, 3))
    data.obs['donor'] = ['a', 'b'] * 30
    path = tmp_path / 'counts.h5ad'
    data.write_h5ad(path)
    return path, data


@pytest.mark.parametrize('method', ['random', 'scsampler'])
def test_draws(counts, tmp_path, method):
    path, source = counts
    manifests = []
    for name in ('first', 'second'):
        obj = cNMF(output_dir=str(tmp_path), name=name)
        manifest_path = obj.sample(path, method, 2, 0.25, 42)
        with open(manifest_path) as stream:
            manifest = json.load(stream)
        folder = tmp_path / name / 'sampling'
        draws = []
        assert manifest['sample_prefix'] == 'counts_sample'
        for number, draw in enumerate(manifest['samples'], 1):
            assert draw['counts'] == f'counts_sample_{number:03d}.h5ad'
            assert draw['cell_indices'] == f'counts_sample_{number:03d}.cell_indices.npy'
            indices = np.load(folder / draw['cell_indices'])
            result = ad.read_h5ad(folder / draw['counts'])
            assert result.shape == (15, 8)
            assert len(np.unique(indices)) == 15
            np.testing.assert_array_equal(result.X.toarray(), source.X[indices].toarray())
            assert result.obs.equals(source.obs.iloc[indices])
            assert result.var.equals(source.var)
            draws.append(indices)
        assert len({d['sample_seed'] for d in manifest['samples']}) == 2
        manifests.append(draws)
    for left, right in zip(*manifests):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize('fraction', [None, 0, -0.1, 1.1, 0.001])
def test_invalid_fraction(counts, tmp_path, fraction):
    with pytest.raises(ValueError, match='sample_fraction'):
        sample(counts[0], tmp_path / 'out', sample_fraction=fraction)


def test_full_sample_and_no_overwrite(counts, tmp_path):
    output = tmp_path / 'out'
    sample(counts[0], output, sample_fraction=1)
    np.testing.assert_array_equal(np.load(output / 'counts_sample_001.cell_indices.npy'), np.arange(60))
    with pytest.raises(FileExistsError):
        sample(counts[0], output, sample_fraction=1)


@pytest.mark.parametrize('matrix_type', ['dense', 'csr', 'csc'])
def test_computed_embedding(counts, tmp_path, monkeypatch, matrix_type):
    monkeypatch.setattr(sampling, 'SAMPLE_PCA_COMPONENTS', 3)
    monkeypatch.setattr(sampling, 'SAMPLE_PCA_CHUNK_SIZE', 29)
    path, data = counts
    del data.obsm['X_pca']
    dense = data.X.toarray()
    dense[0] = 0  # An empty cell must not produce NaNs during normalization.
    data.X = (dense if matrix_type == 'dense'
              else getattr(sparse, matrix_type + '_matrix')(dense))
    data.write_h5ad(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    # Independent small in-memory oracle: CP10K/log1p, then sklearn IPCA.
    # A 2-row tail is merged: batches contain 29 and 31 cells, for 3 PCs.
    prepared = dense.astype(np.float64)
    totals = prepared.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    prepared = np.log1p(prepared * (10000.0 / totals))
    reference = IncrementalPCA(n_components=3)
    reference.partial_fit(prepared[:29])
    reference.partial_fit(prepared[29:])
    expected = reference.transform(prepared).astype(np.float32)

    runs = []
    for name in ('first', 'repeat'):
        manifest_path = cNMF(str(tmp_path), name).sample(
            path, 'scsampler', 2, 0.25, 42)
        folder = tmp_path / name / 'sampling'
        manifest = json.loads(Path(manifest_path).read_text())
        info = manifest['pca']
        assert info['source'] == 'computed'
        assert info['mode'] == 'auto' and info['input'] == 'counts'
        assert info['preprocessing'] == 'log1p(CP10K)'
        assert info['components'] == 3
        assert info['chunk_size'] == 29
        assert info['working_dtype'] == 'float64'
        assert info['gene_selection'] == 'all'
        embedding = np.load(folder / info['embedding'], mmap_mode='r')
        assert isinstance(embedding, np.memmap)
        assert embedding.dtype == np.float32
        assert np.isfinite(embedding).all()
        np.testing.assert_allclose(embedding, expected, rtol=1e-6, atol=1e-6)
        with np.load(folder / info['model']) as model:
            np.testing.assert_allclose(model['mean'], reference.mean_, atol=1e-12)
            np.testing.assert_allclose(model['components'], reference.components_,
                                       rtol=1e-10, atol=1e-10)
        indices = []
        for draw in manifest['samples']:
            selected = np.load(folder / draw['cell_indices'])
            result = ad.read_h5ad(folder / draw['counts'])
            result_x = result.X.toarray() if sparse.issparse(result.X) else result.X
            np.testing.assert_array_equal(result_x, dense[selected])
            assert result.obs.equals(data.obs.iloc[selected])
            assert result.var.equals(data.var)
            indices.append(selected)
        runs.append((np.asarray(embedding).copy(), indices))
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    for first, repeat in zip(runs[0][1], runs[1][1]):
        np.testing.assert_array_equal(first, repeat)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('embedding_key', ['X_pca', 'X_scVI'])
def test_existing_embedding_reused_once(counts, tmp_path, monkeypatch, embedding_key):
    import scsampler
    import sklearn.decomposition

    path, data = counts
    expected = data.obsm['X_pca'].copy()
    if embedding_key != 'X_pca':
        data.obsm[embedding_key] = expected
        data.obsm['X_pca'][:] = np.nan  # Only the requested embedding matters.
        data.write_h5ad(path)

    def unexpected_fit(*args, **kwargs):
        pytest.fail('Existing embedding should be reused without fitting')

    monkeypatch.setattr(sklearn.decomposition, 'IncrementalPCA', unexpected_fit)
    actual_sampler = scsampler.scsampler
    received = []

    def record_embedding(embedding, **kwargs):
        np.testing.assert_array_equal(embedding, expected)
        received.append(embedding)
        return actual_sampler(embedding, **kwargs)

    monkeypatch.setattr(scsampler, 'scsampler', record_embedding)
    path = sample(path, tmp_path / 'out', 'scsampler', 3, 0.25,
                  sample_embedding_key=embedding_key)
    manifest = json.loads(Path(path).read_text())
    assert manifest['pca']['source'] == f'obsm/{embedding_key}'
    assert manifest['pca']['components'] == 3  # Keep existing PCs, not default 30.
    assert manifest['pca']['dtype'] == 'float64'
    assert manifest['pca']['preprocessing'] is None
    assert len(received) == 3 and all(x is received[0] for x in received)
    assert not (tmp_path / 'out' / 'pca.npy').exists()


@pytest.mark.parametrize('method,fraction', [('random', 0.5), ('scsampler', 1.0)])
def test_skip_pca_when_unneeded(counts, tmp_path, monkeypatch, method, fraction):
    path, data = counts
    del data.obsm['X_pca']
    data.write_h5ad(path)

    def unexpected_pca(*args, **kwargs):
        pytest.fail('PCA is unnecessary for random sampling or selecting all cells')

    monkeypatch.setattr(sampling, '_sample_embedding', unexpected_pca)
    manifest_path = sample(path, tmp_path / 'out', method, sample_fraction=fraction)
    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest['pca'] is None
    assert not (tmp_path / 'out' / 'pca.npy').exists()


@pytest.mark.parametrize('via_cli', [False, True])
def test_custom_embedding_key(counts, tmp_path, monkeypatch, via_cli):
    import scsampler

    path, data = counts
    data.obsm['X_scVI'] = np.random.default_rng(321).normal(size=(data.n_obs, 2))
    data.obsm['X_pca'][:] = np.nan
    data.write_h5ad(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    received = []

    def record_embedding(embedding, n_obs, **kwargs):
        np.testing.assert_array_equal(embedding, data.obsm['X_scVI'])
        received.append(embedding)
        return np.arange(n_obs)

    monkeypatch.setattr(scsampler, 'scsampler', record_embedding)
    if via_cli:
        monkeypatch.setattr('sys.argv', [
            'cnmf', 'sample', '--counts', str(path),
            '--output-dir', str(tmp_path), '--name', 'custom_embedding',
            '--sample-method', 'scsampler', '--sample-fraction', '0.25',
            '--sample-embedding-key', 'X_scVI',
        ])
        main()
    else:
        cNMF(str(tmp_path), 'custom_embedding').sample(
            path, 'scsampler', sample_fraction=0.25, sample_embedding_key='X_scVI')
    output = tmp_path / 'custom_embedding' / 'sampling'
    info = json.loads((output / 'manifest.json').read_text())['pca']
    assert len(received) == 1
    assert info['source'] == 'obsm/X_scVI' and info['components'] == 2
    assert info['preprocessing'] is None
    assert not (output / 'pca.npy').exists()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_missing_custom_embedding_does_not_fall_back(counts, tmp_path, monkeypatch):
    import sklearn.decomposition

    def unexpected_fit(*args, **kwargs):
        pytest.fail('A missing custom embedding must not trigger PCA')

    monkeypatch.setattr(sklearn.decomposition, 'IncrementalPCA', unexpected_fit)
    with pytest.raises(ValueError, match='X_scVI'):
        sample(counts[0], tmp_path / 'out', 'scsampler', sample_fraction=0.5,
               sample_embedding_key='X_scVI')


def test_random_ignores_embedding_key(counts, tmp_path):
    manifests = []
    for name, embedding_key in [('default', 'X_pca'), ('missing', 'X_missing')]:
        path = sample(counts[0], tmp_path / name, sample_fraction=0.5,
                      sample_embedding_key=embedding_key)
        manifest = json.loads(Path(path).read_text())
        assert manifest['pca'] is None
        manifests.append(np.load(Path(path).parent / manifest['samples'][0]['cell_indices']))
    np.testing.assert_array_equal(*manifests)


@pytest.mark.parametrize('shape,components', [((60, 8), 8), ((5, 40), 4), ((40, 40), 30)])
def test_pca_internal_defaults_clamp_to_input_size(tmp_path, shape, components):
    data = ad.AnnData(np.random.default_rng(143).poisson(3, shape))
    embedding, info = _sample_embedding(data, tmp_path)
    assert sampling.SAMPLE_PCA_COMPONENTS == 30
    assert sampling.SAMPLE_PCA_CHUNK_SIZE == 2000
    assert embedding.shape == (shape[0], components)
    assert info['components'] == components
    assert info['chunk_size'] == 2000
    assert np.isfinite(embedding).all()


def test_sampling_help_exposes_only_embedding_option(monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['cnmf', '--help'])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert '--sample-embedding-key' in help_text
    assert '--sample-pca-' not in help_text


@pytest.mark.parametrize('option,value', [
    ('mode', 'auto'), ('components', '3'), ('chunk-size', '29'), ('input', 'counts'),
])
def test_old_pca_cli_options_removed(counts, tmp_path, monkeypatch, capsys, option, value):
    monkeypatch.setattr('sys.argv', [
        'cnmf', 'sample', '--counts', str(counts[0]), '--output-dir', str(tmp_path),
        '--name', 'old_pca', '--sample-fraction', '0.5',
        f'--sample-pca-{option}', value,
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert f'unrecognized arguments: --sample-pca-{option}' in capsys.readouterr().err


def test_pca_reads_only_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(sampling, 'SAMPLE_PCA_COMPONENTS', 3)
    monkeypatch.setattr(sampling, 'SAMPLE_PCA_CHUNK_SIZE', 29)
    # A guarded array rejects whole-matrix conversion and records both passes.
    values = np.random.default_rng(123).poisson(3, (60, 8))
    before = values.copy()
    reads = []

    class ChunkOnlyMatrix:
        def __array__(self, *args, **kwargs):
            pytest.fail('PCA must not materialize the complete matrix')

        def __getitem__(self, rows):
            assert isinstance(rows, slice)
            assert rows.start is not None and rows.stop is not None
            assert rows.stop - rows.start <= 31
            reads.append((rows.start, rows.stop))
            return values[rows]

    data = SimpleNamespace(X=ChunkOnlyMatrix(), n_obs=60, n_vars=8, obsm={})
    embedding, info = _sample_embedding(data, tmp_path)
    assert reads == [(0, 29), (29, 60), (0, 29), (29, 60)]
    assert isinstance(embedding, np.memmap) and not embedding.flags.writeable
    assert embedding.shape == (60, 3)
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_invalid_existing_embedding(counts, tmp_path, value):
    path, data = counts
    data.obsm['X_pca'][0, 0] = value
    data.write_h5ad(path)
    with pytest.raises(ValueError, match='X_pca.*NaN or Inf'):
        sample(path, tmp_path / 'out', 'scsampler', sample_fraction=0.5)


@pytest.mark.parametrize('embedding', [
    np.ones((60, 0)), np.ones((59, 2)), np.ones((60, 2, 1)),
    np.full((60, 2), 'text'), np.ones((60, 2), dtype=np.complex128),
])
def test_invalid_embedding_shape_or_dtype(tmp_path, embedding):
    # Namespace bypasses AnnData's own shape validation to test our contract.
    data = SimpleNamespace(n_obs=60, obsm={'X_scVI': embedding})
    with pytest.raises(ValueError, match='X_scVI'):
        _sample_embedding(data, tmp_path, sample_embedding_key='X_scVI')


@pytest.mark.parametrize('value,match', [
    (-1, 'nonnegative'), (np.nan, 'NaN or Inf'), (np.inf, 'NaN or Inf'),
])
def test_invalid_counts_for_pca(counts, tmp_path, value, match):
    path, data = counts
    del data.obsm['X_pca']
    data.X = data.X.toarray().astype(np.float64)
    data.X[0, 0] = value
    data.write_h5ad(path)
    with pytest.raises(ValueError, match=match):
        sample(path, tmp_path / 'out', 'scsampler', sample_fraction=0.5)


@pytest.mark.parametrize('via_cli', [False, True])
def test_custom_prefix(counts, tmp_path, monkeypatch, via_cli):
    if via_cli:
        monkeypatch.setattr('sys.argv', [
            'cnmf', 'sample', '--counts', str(counts[0]),
            '--output-dir', str(tmp_path), '--name', 'custom',
            '--sample-fraction', '0.5', '--sample-times', '2',
            '--sample-prefix', 'pbmc_half',
        ])
        main()
    else:
        cNMF(output_dir=str(tmp_path), name='custom').sample(
            counts[0], sample_fraction=0.5, sample_times=2,
            sample_prefix='pbmc_half')
    output = tmp_path / 'custom' / 'sampling'
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['sample_prefix'] == 'pbmc_half'
    for number, draw in enumerate(manifest['samples'], 1):
        assert draw['counts'] == f'pbmc_half_{number:03d}.h5ad'
        assert draw['cell_indices'] == f'pbmc_half_{number:03d}.cell_indices.npy'
        assert (output / draw['counts']).is_file()
        assert (output / draw['cell_indices']).is_file()


@pytest.mark.parametrize('prefix', ['', '.', '..', '../escape', '/tmp/escape', 'sub/name'])
def test_invalid_prefix(counts, tmp_path, prefix):
    with pytest.raises(ValueError, match='sample_prefix'):
        sample(counts[0], tmp_path / 'out', sample_fraction=0.5,
               sample_prefix=prefix)
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('matrix_type', ['dense', 'csr', 'csc'])
@pytest.mark.parametrize('dtype', [np.int64, np.float32, np.float64])
def test_full_input_hvgs_match_cnmf(tmp_path, matrix_type, dtype):
    # Deterministic synthetic counts with unequal expression/dispersion.
    # The oracle is the existing in-memory cNMF TPM + HVG implementation,
    # using ALL 143 cells, not any of the sampled subsets.
    rng = np.random.default_rng(809)
    values = rng.negative_binomial(
        3, rng.uniform(0.15, 0.85, 67), size=(143, 67)).astype(dtype)
    values[0] = 0
    values[:, 0] = 0
    matrix = (values if matrix_type == 'dense'
              else getattr(sparse, matrix_type + '_matrix')(values))
    data = ad.AnnData(matrix)
    data.var_names = [f'gene_{66 - i}' for i in range(67)]
    path = tmp_path / 'full.h5ad'
    data.write_h5ad(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    tpm = compute_tpm(data)
    calculate = get_highvar_genes_sparse if sparse.issparse(tpm.X) else get_highvar_genes
    expected, _ = calculate(tpm.X, numgenes=13)
    expected_genes = data.var_names[expected.high_var].tolist()

    outputs = []
    for chunk_size in (17, 61):
        folder = tmp_path / f'chunk_{chunk_size}'
        folder.mkdir()
        backed = ad.read_h5ad(path, backed='r')
        try:
            info = _sample_hvgs(backed, folder, 'full_sample', 13,
                                sample_hvg_chunk_size=chunk_size)
        finally:
            backed.file.close()
        genes = (folder / info['genes_file']).read_text().splitlines()
        assert genes == expected_genes
        assert info['source'] == 'computed'
        assert info['method'] == 'cnmf_fano_ratio'
        assert info['n_input_cells'] == 143
        assert info['n_genes'] == 13
        assert info['normalization'] == 'CPM (target_sum=1e6)'
        outputs.append(genes)
    assert outputs[0] == outputs[1]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('method', ['random', 'scsampler'])
def test_hvgs_once_then_prepare_reuses_list(counts, tmp_path, monkeypatch, method):
    path, data = counts
    expected, _ = get_highvar_genes_sparse(compute_tpm(data).X, numgenes=4)
    expected_genes = data.var_names[expected.high_var].tolist()
    original = cnmf_module.get_highvar_genes_from_stats
    calls = []

    def record_ranking(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(cnmf_module, 'get_highvar_genes_from_stats', record_ranking)
    manifest_path = cNMF(str(tmp_path), 'sampled').sample(
        path, method, 3, 0.5, sample_numgenes=4, sample_hvg_chunk_size=17,
        sample_export_chunk_size=7)
    folder = Path(manifest_path).parent
    manifest = json.loads(Path(manifest_path).read_text())
    genes_file = folder / manifest['hvg']['genes_file']
    assert genes_file.read_text().splitlines() == expected_genes
    assert len(calls) == 1  # One reference for all three draws.

    def unexpected_selection(*args, **kwargs):
        pytest.fail('prepare --genes-file must not calculate HVGs again')

    monkeypatch.setattr(cnmf_module, 'get_highvar_genes', unexpected_selection)
    monkeypatch.setattr(cnmf_module, 'get_highvar_genes_sparse', unexpected_selection)
    for i, draw in enumerate(manifest['samples']):
        run = cNMF(str(tmp_path), f'prepared_{i}')
        run.prepare(str(folder / draw['counts']), [2], n_iter=2,
                    genes_file=str(genes_file), seed=3)
        normalized = ad.read_h5ad(run.paths['normalized_counts'])
        assert normalized.var_names.tolist() == expected_genes
        assert Path(run.paths['nmf_genes_list']).read_text().splitlines() == expected_genes
    assert len(calls) == 1


def test_existing_hvgs_reused_without_calculation(counts, tmp_path, monkeypatch):
    path, data = counts
    data.var['highly_variable'] = [False, True, False, False, True, False, False, False]
    data.write_h5ad(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    def unexpected_calculation(*args, **kwargs):
        pytest.fail('An existing HVG mask must be reused without recalculation')

    monkeypatch.setattr(cnmf_module, 'compute_tpm', unexpected_calculation)
    manifest_path = sample(path, tmp_path / 'out', sample_times=3,
                           sample_fraction=0.5, sample_numgenes=4)
    folder = Path(manifest_path).parent
    info = json.loads(Path(manifest_path).read_text())['hvg']
    assert info['source'] == 'var/highly_variable'
    assert info['method'] is None  # Do not invent provenance for an existing mask.
    assert info['n_genes'] == 2  # Preserve the existing mask, not requested 4.
    assert (folder / info['genes_file']).read_text().splitlines() == ['1', '4']
    for result in folder.glob('*.h5ad'):
        assert ad.read_h5ad(result).var.equals(data.var)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_hvg_options_via_cli(counts, tmp_path, monkeypatch):
    path, data = counts
    # Recompute must ignore even an invalid pre-existing mask.
    data.var['highly_variable'] = np.nan
    data.write_h5ad(path)
    monkeypatch.setattr('sys.argv', [
        'cnmf', 'sample', '--counts', str(path),
        '--output-dir', str(tmp_path), '--name', 'hvg_cli',
        '--sample-fraction', '0.5', '--sample-times', '2',
        '--sample-prefix', 'testing_data', '--sample-numgenes', '3',
        '--sample-hvg-mode', 'recompute', '--sample-hvg-chunk-size', '17',
    ])
    main()
    folder = tmp_path / 'hvg_cli' / 'sampling'
    info = json.loads((folder / 'manifest.json').read_text())['hvg']
    expected, _ = get_highvar_genes_sparse(compute_tpm(data).X, numgenes=3)
    assert info['source'] == 'computed' and info['mode'] == 'recompute'
    assert info['n_genes'] == 3 and info['chunk_size'] == 17
    assert info['genes_file'] == 'testing_data.overdispersed_genes.txt'
    assert (folder / info['genes_file']).read_text().splitlines() == data.var_names[expected.high_var].tolist()


def test_hvg_reads_only_chunks(tmp_path):
    # Guard against a full-X materialization; also prove one statistics pass.
    values = np.random.default_rng(54).poisson(3, (60, 8))
    before = values.copy()
    reads = []

    class ChunkOnlyMatrix:
        def __array__(self, *args, **kwargs):
            pytest.fail('HVG calculation must not materialize the full matrix')

        def __getitem__(self, rows):
            assert isinstance(rows, slice)
            assert rows.stop - rows.start <= 17
            reads.append((rows.start, rows.stop))
            return values[rows]

    names = pd.Index([f'gene_{i}' for i in range(8)])
    data = SimpleNamespace(X=ChunkOnlyMatrix(), n_obs=60, var=pd.DataFrame(index=names),
                           var_names=names)
    info = _sample_hvgs(data, tmp_path, 'guarded', 4, sample_hvg_chunk_size=17)
    assert reads == [(0, 17), (17, 34), (34, 51), (51, 68)]
    assert info['n_genes'] == 4
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize('with_tpm', [False, True])
def test_recomputed_hvgs_exclude_constant_genes(tmp_path, with_tpm):
    values = np.random.default_rng(61).poisson(3, (60, 12)).astype(float)
    values[:, 0] = 0
    data = ad.AnnData(sparse.csr_matrix(values))
    tpm = None
    if with_tpm:
        tpm = compute_tpm(data)
        expression = tpm.X.toarray()
        expression[:, 1] = 5  # Counts vary, but the supplied TPM gene is constant.
        tpm.X = sparse.csr_matrix(expression)
    info = _sample_hvgs(data, tmp_path, 'sample', 20,
                        sample_hvg_mode='recompute', sample_hvg_chunk_size=17, tpm=tpm)
    genes = (tmp_path / info['genes_file']).read_text().splitlines()
    assert genes == data.var_names[2 if with_tpm else 1:].tolist()


def test_skip_hvg_calculation(counts, tmp_path, monkeypatch):
    def unexpected_calculation(*args, **kwargs):
        pytest.fail('sample_numgenes=0 must skip HVG calculation')

    monkeypatch.setattr(cnmf_module, 'compute_tpm', unexpected_calculation)
    path = sample(counts[0], tmp_path / 'out', sample_fraction=0.5, sample_numgenes=0)
    assert json.loads(Path(path).read_text())['hvg'] is None
    assert not list(Path(path).parent.glob('*.overdispersed_genes.txt'))


@pytest.mark.parametrize('options,match', [
    ({'sample_numgenes': -1}, 'sample_numgenes'),
    ({'sample_numgenes': 1.5}, 'sample_numgenes'),
    ({'sample_hvg_mode': 'invalid'}, 'sample_hvg_mode'),
    ({'sample_hvg_chunk_size': 0}, 'sample_hvg_chunk_size'),
    ({'sample_hvg_chunk_size': 1.5}, 'sample_hvg_chunk_size'),
])
def test_invalid_hvg_options(counts, tmp_path, options, match):
    with pytest.raises(ValueError, match=match):
        sample(counts[0], tmp_path / 'out', sample_fraction=0.5, **options)


@pytest.mark.parametrize('mask,match', [
    ([0, 1] * 4, 'boolean'),
    ([True, None] * 4, 'boolean'),
    ([False] * 8, 'empty'),
])
def test_invalid_hvg_mask(counts, tmp_path, mask, match):
    path, data = counts
    # Validate directly so H5AD serialization does not reject malformed flags first.
    data.var['highly_variable'] = mask
    with pytest.raises(ValueError, match=match):
        _sample_hvgs(data, tmp_path, 'invalid')


@pytest.mark.parametrize('numgenes,threshold', [(4, None), (None, None), (None, 1.5)])
def test_hvg_helper_extraction_preserves_ranking(numgenes, threshold):
    # The unchanged dense cNMF function is an independent oracle for the
    # sparse function whose Fano-ranking body was extracted for streaming reuse.
    values = np.random.default_rng(764).poisson(3, (80, 40)).astype(np.float64)
    expected, expected_params = get_highvar_genes(
        values, expected_fano_threshold=threshold, numgenes=numgenes)
    actual, actual_params = get_highvar_genes_sparse(
        sparse.csr_matrix(values), expected_fano_threshold=threshold, numgenes=numgenes)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)
    for key in expected_params:
        if expected_params[key] is None:
            assert actual_params[key] is None
        else:
            assert actual_params[key] == pytest.approx(expected_params[key], rel=1e-12)


@pytest.mark.parametrize('matrix_type', ['dense', 'csr', 'csc'])
@pytest.mark.parametrize('dtype', [np.int64, np.float32, np.float64])
@pytest.mark.parametrize('chunk_size', [1, 7])
@pytest.mark.parametrize('fraction', [0.7, 1.0])
def test_chunk_export_preserves_values_and_annotations(
        tmp_path, matrix_type, dtype, chunk_size, fraction):
    # Deterministic synthetic AnnData with extra raw genes, mixed layers and
    # graph edges spanning export chunks. Oracle: the original in-memory slice.
    rng = np.random.default_rng(204)

    def matrix(values):
        return (values if matrix_type == 'dense'
                else getattr(sparse, matrix_type + '_matrix')(values))

    values = rng.poisson(0.6, (23, 9)).astype(dtype)
    values[0] = 0
    values[:, 0] = 0
    if np.issubdtype(dtype, np.floating):
        values[1, 1] = np.nextafter(dtype(0), dtype(1))  # Must not threshold.
    source = ad.AnnData(matrix(values),
                       obs=pd.DataFrame(index=[f'cell_{i}' for i in range(23)]),
                       var=pd.DataFrame(index=[f'gene_{i}' for i in range(9)]))
    source.obs['donor'] = pd.Categorical(['a', 'b'] * 11 + ['a'],
                                        categories=['a', 'b', 'unused'])
    source.var['highly_variable'] = [False, True, True] * 3
    source.layers['counts'] = sparse.csr_matrix(values)
    source.layers['dense'] = values + dtype(1)
    source.obsm['X_pca'] = rng.normal(size=(23, 3)).astype(np.float32)
    source.obsm['frame'] = pd.DataFrame(rng.normal(size=(23, 2)),
                                       index=source.obs_names, columns=['a', 'b'])
    source.varm['loadings'] = rng.normal(size=(9, 3))
    source.varp['genes'] = sparse.eye(9, format='csr')
    source.obsp['graph'] = sparse.csr_matrix(rng.integers(0, 3, (23, 23)))
    source.obsp['csc_graph'] = source.obsp['graph'].tocsc()
    source.obsp['dense_graph'] = source.obsp['graph'].toarray()
    source.uns['settings'] = {'seed': 204, 'names': np.array(['a', 'b'])}
    raw = ad.AnnData(matrix(rng.poisson(0.6, (23, 13)).astype(dtype)),
                     obs=source.obs.copy(),
                     var=pd.DataFrame({'kind': ['raw'] * 13},
                                      index=[f'raw_gene_{i}' for i in range(13)]))
    raw.varm['loadings'] = rng.normal(size=(13, 2))
    source.raw = raw
    path = tmp_path / 'annotated.h5ad'
    source.write_h5ad(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    manifest_path = sample(path, tmp_path / 'out', sample_fraction=fraction,
                           sample_numgenes=0, sample_export_chunk_size=chunk_size)
    folder = Path(manifest_path).parent
    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest['export'] == {'chunk_size': chunk_size, 'sparse': True}
    draw = manifest['samples'][0]
    indices = np.load(folder / draw['cell_indices'])
    expected = ad.read_h5ad(path)[indices].to_memory()
    actual = ad.read_h5ad(folder / draw['counts'])

    def same_matrix(left, right):
        assert left.dtype == right.dtype
        left = left.toarray() if sparse.issparse(left) else left
        right = right.toarray() if sparse.issparse(right) else right
        np.testing.assert_array_equal(left, right)

    assert sparse.issparse(actual.X)
    assert actual.X.format == ('csc' if matrix_type == 'csc' else 'csr')
    same_matrix(actual.X, expected.X)
    pd.testing.assert_frame_equal(actual.obs, expected.obs)
    pd.testing.assert_frame_equal(actual.var, expected.var)
    for key in ('layers', 'obsm', 'varm', 'obsp', 'varp'):
        actual_mapping, expected_mapping = getattr(actual, key), getattr(expected, key)
        assert actual_mapping.keys() == expected_mapping.keys()
        for name in expected_mapping:
            if isinstance(expected_mapping[name], pd.DataFrame):
                pd.testing.assert_frame_equal(actual_mapping[name], expected_mapping[name].copy())
            else:
                same_matrix(actual_mapping[name], expected_mapping[name])
    assert actual.uns['settings']['seed'] == expected.uns['settings']['seed']
    np.testing.assert_array_equal(actual.uns['settings']['names'],
                                  expected.uns['settings']['names'])
    same_matrix(actual.raw.X, expected.raw.X)
    pd.testing.assert_frame_equal(actual.raw.var, expected.raw.var)
    np.testing.assert_array_equal(actual.raw.varm['loadings'], expected.raw.varm['loadings'])
    with h5py.File(folder / draw['counts'], 'r') as handle:
        assert handle['X/indptr'].dtype == np.dtype('int64')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not [path for path in folder.iterdir() if path.name.startswith('.')]


@pytest.mark.parametrize('matrix_type', ['dense', 'csr', 'csc'])
@pytest.mark.parametrize('pairwise', [False, True])
def test_export_reads_only_bounded_matrix_chunks(tmp_path, matrix_type, pairwise):
    n_rows, n_cols, chunk_size = 53, 41, 7
    if pairwise:
        n_cols = n_rows
    values = np.arange(n_rows * n_cols, dtype=np.float32).reshape(n_rows, n_cols)
    matrix = (values if matrix_type == 'dense'
              else getattr(sparse, matrix_type + '_matrix')(values))
    indices = np.arange(1, n_rows, 2, dtype=np.int64)
    reads = []

    class ChunkOnlyMatrix:
        format = matrix_type
        shape = matrix.shape
        dtype = matrix.dtype

        def __array__(self, *args, **kwargs):
            pytest.fail('Export must not materialize the full matrix')

        def __getitem__(self, key):
            rows, cols = key
            if matrix_type == 'csc':
                assert rows == slice(None)
                count = (cols.stop - cols.start if isinstance(cols, slice) else len(cols))
                assert count <= max(1, chunk_size * n_cols // n_rows)
            else:
                assert cols == slice(None)
                assert len(rows) <= chunk_size
            reads.append(key)
            return matrix[key]

    path = tmp_path / 'guarded.h5ad'
    width = len(indices) if pairwise else n_cols
    ad.AnnData(obs=pd.DataFrame(index=indices.astype(str)),
               var=pd.DataFrame(index=np.arange(width).astype(str))).write_h5ad(path)
    with h5py.File(path, 'a') as handle:
        sampling._write_sample_matrix(handle, 'X', ChunkOnlyMatrix(),
                                       indices, chunk_size, pairwise=pairwise)
    result = ad.read_h5ad(path).X.toarray()
    expected = values[indices][:, indices] if pairwise else values[indices]
    np.testing.assert_array_equal(result, expected)
    assert len(reads) > 1


@pytest.mark.parametrize('matrix_type', ['dense', 'csr', 'csc'])
def test_export_never_materializes_backed_matrix(counts, tmp_path, monkeypatch, matrix_type):
    path, source = counts
    values = source.X.toarray()
    source.X = (values if matrix_type == 'dense'
                else getattr(sparse, matrix_type + '_matrix')(values))
    source.write_h5ad(path)

    def unexpected_memory(*args, **kwargs):
        pytest.fail('No full AnnData/matrix materialization during export')

    monkeypatch.setattr(ad.AnnData, 'to_memory', unexpected_memory)
    backed = ad.read_h5ad(path, backed='r')
    try:
        if matrix_type != 'dense':
            monkeypatch.setattr(type(backed.X), 'to_memory', unexpected_memory)
    finally:
        backed.file.close()
    manifest_path = sample(path, tmp_path / 'out', sample_fraction=0.65,
                           sample_numgenes=0, sample_export_chunk_size=7)
    monkeypatch.undo()  # The verification read below intentionally loads the output.
    folder = Path(manifest_path).parent
    indices = np.load(folder / 'counts_sample_001.cell_indices.npy')
    result = ad.read_h5ad(folder / 'counts_sample_001.h5ad')
    np.testing.assert_array_equal(result.X.toarray(), values[indices])


def test_export_option_via_cli(counts, tmp_path, monkeypatch):
    monkeypatch.setattr('sys.argv', [
        'cnmf', 'sample', '--counts', str(counts[0]),
        '--output-dir', str(tmp_path), '--name', 'export_cli',
        '--sample-fraction', '0.5', '--sample-export-chunk-size', '7',
    ])
    main()
    folder = tmp_path / 'export_cli' / 'sampling'
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['export']['chunk_size'] == 7
    assert sparse.issparse(ad.read_h5ad(folder / manifest['samples'][0]['counts']).X)


@pytest.mark.parametrize('chunk_size', [0, -1, 1.5])
def test_invalid_export_chunk_size(counts, tmp_path, chunk_size):
    with pytest.raises(ValueError, match='sample_export_chunk_size'):
        sample(counts[0], tmp_path / 'out', sample_fraction=0.5,
               sample_export_chunk_size=chunk_size)
    assert not (tmp_path / 'out').exists()


def test_failed_export_cleans_partial_file(counts, tmp_path, monkeypatch):
    original = sampling._write_sample_matrix

    def interrupted_write(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError('simulated write failure')

    monkeypatch.setattr(sampling, '_write_sample_matrix', interrupted_write)
    with pytest.raises(OSError, match='simulated write failure'):
        sample(counts[0], tmp_path / 'out', sample_fraction=0.5,
               sample_numgenes=0, sample_export_chunk_size=7)
    assert list((tmp_path / 'out').iterdir()) == []

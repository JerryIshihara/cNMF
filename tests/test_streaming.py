"""Deterministic synthetic parity against sklearn fixed-factor CD, no downloads."""
import anndata as ad
import numpy as np
import pytest
from scipy import sparse
from sklearn.decomposition import non_negative_factorization
from cnmf.streaming import refit_usage_streaming, refit_spectra_streaming


@pytest.mark.parametrize('storage', ['dense', 'csr', 'backed'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_streamed_qc(tmp_path, monkeypatch, storage, dtype):
    import cnmf.streaming as streaming
    from cnmf.qc import get_r2
    rng = np.random.default_rng(9)
    x = rng.random((37, 19)).astype(dtype)
    w = rng.random((37, 3)).astype(dtype)
    h = rng.random((3, 19)).astype(dtype)
    expected = get_r2(x.astype(float), w.astype(float), h.astype(float))
    source = sparse.csr_matrix(x) if storage == 'csr' else x
    if storage == 'backed':
        source = tmp_path / 'qc.h5ad'
        ad.AnnData(sparse.csr_matrix(x)).write_h5ad(source)
    monkeypatch.setattr(streaming, '_chunk_size', lambda *args: 5)
    actual = streaming.reconstruction_stats_streaming(source, w, h)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_auto_chunk_budget():
    from cnmf.streaming import _chunk_size, MEMORY_BUDGET, DEFAULT_WORKERS
    rows = _chunk_size((8_000_000, 20_000), 20, 1_280_000_000)
    assert 1 < rows < 8_000_000
    assert DEFAULT_WORKERS * rows * 8 * (6*20_000 + 4*20) + 1_280_000_000 <= MEMORY_BUDGET * .75
    assert _chunk_size((10, 20), 3) == 3
    with pytest.raises(ValueError, match='one row'):
        _chunk_size((10, 20), 3, MEMORY_BUDGET)


def test_default_threads(tmp_path, monkeypatch):
    import cnmf.streaming as streaming
    from threadpoolctl import threadpool_info
    original = streaming.non_negative_factorization
    observed = []
    def record(*args, **kwargs):
        observed.extend(p['num_threads'] for p in threadpool_info() if p['user_api'] == 'blas')
        return original(*args, **kwargs)
    monkeypatch.setattr(streaming, 'non_negative_factorization', record)
    refit_usage_streaming(np.ones((10, 5)), np.ones((2, 5)), tmp_path / 'w.npy')
    assert observed and set(observed) == {4}


def test_four_chunk_workers_bounded_and_ordered(monkeypatch):
    import threading
    import cnmf.streaming as streaming
    barrier = threading.Barrier(4)
    thread_ids = set()
    reads = []
    reader_thread = threading.get_ident()
    original = streaming._blocks
    def blocks(x, rows):
        for item in original(x, rows):
            assert threading.get_ident() == reader_thread
            reads.append(item[0])
            yield item
    def work(start, stop, block):
        thread_ids.add(threading.get_ident())
        barrier.wait(timeout=10)
        return start
    monkeypatch.setattr(streaming, '_blocks', blocks)
    results = streaming._map_blocks(work, np.ones((8, 2)), 1)
    assert next(results) == 0
    assert reads == [0, 1, 2, 3]
    assert list(results) == list(range(1, 8))
    assert len(thread_ids) == 4


def test_worker_count_parity(tmp_path, monkeypatch):
    import cnmf.streaming as streaming
    rng = np.random.default_rng(18)
    x, w, h = rng.random((41, 17)), rng.random((41, 3)), rng.random((3, 17))
    results = []
    for workers in [1, 4]:
        monkeypatch.setattr(streaming, 'DEFAULT_WORKERS', workers)
        wp, hp = tmp_path / ('w%d.npy' % workers), tmp_path / ('h%d.npy' % workers)
        refit_usage_streaming(x, h, wp, chunk_size=5)
        refit_spectra_streaming(x, w, hp, chunk_size=5)
        monkeypatch.setattr(streaming, '_chunk_size', lambda *args: 5)
        results.append((np.load(wp), np.load(hp), streaming.reconstruction_stats_streaming(x, w, h)))
    for single, parallel in zip(*results):
        np.testing.assert_array_equal(single, parallel)


@pytest.mark.parametrize('storage', ['npy', 'dense', 'csr'])
@pytest.mark.parametrize('chunk_size', [7, 25, 200])
@pytest.mark.parametrize('alpha', [0., .01])
@pytest.mark.parametrize('input_dtype', [np.float32, np.float64])
def test_refits(tmp_path, storage, chunk_size, alpha, input_dtype):
    rng = np.random.default_rng(42)
    x = rng.gamma(2, 1, (83, 31)).astype(input_dtype)
    w = rng.random((83, 3)).astype(input_dtype)
    h = rng.random((3, 31)).astype(input_dtype)
    x64, w64, h64 = [a.astype(np.float64) for a in (x, w, h)]
    path = tmp_path / ('x.npy' if storage == 'npy' else 'x.h5ad')
    if storage == 'npy':
        np.save(path, x)
    else:
        ad.AnnData(sparse.csr_matrix(x) if storage == 'csr' else x).write_h5ad(path)
    kwargs = dict(solver='cd', update_H=False, max_iter=1000, tol=1e-10,
                  alpha_W=alpha, alpha_H=0., l1_ratio=.4, shuffle=False)
    expected_w, _, _ = non_negative_factorization(x64, H=h64, n_components=3, **kwargs)
    out = tmp_path / 'w.npy'
    refit_usage_streaming(path, h, out, chunk_size, tol=1e-10, alpha=alpha, l1_ratio=.4)
    np.testing.assert_allclose(np.load(out), expected_w, atol=1e-8, rtol=1e-7)
    assert np.load(out).dtype == np.float64
    expected_ht, _, _ = non_negative_factorization(x64.T, H=w64.T, n_components=3, **kwargs)
    out_h = tmp_path / 'h.npy'
    receipt = refit_spectra_streaming(path, w, out_h, chunk_size, tol=1e-10,
                                     alpha=alpha, l1_ratio=.4)
    actual = np.load(out_h)
    np.testing.assert_allclose(actual, expected_ht.T, atol=1e-10, rtol=1e-8)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(receipt['SSE'], np.square(x64-w64@actual).sum(), rtol=1e-12)
    if storage == 'npy':
        np.testing.assert_array_equal(np.load(path), x)
        assert np.load(path, mmap_mode='r').dtype == input_dtype
    with pytest.raises(FileExistsError):
        refit_spectra_streaming(path, w, out_h)


def test_zero_program_and_invalid_input(tmp_path):
    x = np.ones((20, 10))
    w = np.ones((20, 2))
    w[:, 1] = 0
    path = tmp_path / 'x.npy'
    np.save(path, x)
    out = tmp_path / 'h.npy'
    refit_spectra_streaming(path, w, out, chunk_size=3)
    assert np.all(np.load(out)[1] == 0)
    x[-1, -1] = np.nan
    np.save(path, x)
    with pytest.raises(ValueError, match='finite'):
        refit_usage_streaming(path, np.ones((2, 10)), tmp_path / 'bad.npy', chunk_size=3)
    assert not (tmp_path / 'bad.npy').exists()
    assert not list(tmp_path.glob('.refit-*'))


def test_csc_rejected(tmp_path):
    path = tmp_path / 'csc.h5ad'
    ad.AnnData(sparse.csc_matrix(np.ones((10, 5)))).write_h5ad(path)
    with pytest.raises(ValueError, match='CSC'):
        refit_usage_streaming(path, np.ones((2, 5)), tmp_path / 'out.npy')


def test_cnmf_consensus_streaming_parity(tmp_path, monkeypatch):
    import sys
    import pandas as pd
    import yaml
    from cnmf.cnmf import cNMF, main, load_df_from_npz, save_df_to_npz
    rng = np.random.default_rng(73)
    counts = tmp_path / 'counts.h5ad'
    ad.AnnData(rng.poisson(8, (60, 12)).astype(float)).write_h5ad(counts)
    prototypes = rng.random((2, 6)) + .1
    objects = []
    for name in ['original', 'streamed']:
        obj = cNMF(str(tmp_path), name)
        obj.prepare(str(counts), components=[2], n_iter=3, densify=True,
                    seed=14, num_highvar_genes=6, max_NMF_iter=1000,
                    beta_loss='frobenius')
        with open(obj.paths['nmf_run_parameters']) as f:
            options = yaml.safe_load(f)
        options.update(solver='cd', tol=1e-10)
        with open(obj.paths['nmf_run_parameters'], 'w') as f:
            yaml.safe_dump(options, f)
        genes = ad.read_h5ad(obj.paths['normalized_counts']).var_names
        merged = pd.DataFrame(np.vstack([prototypes*.98, prototypes, prototypes*1.02]),
                              index=['iter%d_topic%d' % (i,j) for i in range(3) for j in (1,2)],
                              columns=genes)
        save_df_to_npz(merged, obj.paths['merged_spectra'] % 2)
        objects.append(obj)
    import cnmf.cnmf as cnmf_module
    original_qc = cnmf_module.run_qc
    qc_flags = []
    def record_qc(*args, **kwargs):
        qc_flags.append(kwargs.get('streaming', False))
        return original_qc(*args, **kwargs)
    monkeypatch.setattr(cnmf_module, 'run_qc', record_qc)
    objects[0].consensus(2, density_threshold=2., local_neighborhood_size=.5,
                         show_clustering=False)
    import cnmf.streaming as streaming
    monkeypatch.setattr(streaming, '_chunk_size', lambda *args: 7)
    original = streaming._refit_from_cnmf
    calls = []
    def record(*args, **kwargs):
        calls.append(kwargs.get('spectra', False))
        return original(*args, **kwargs)
    monkeypatch.setattr(streaming, '_refit_from_cnmf', record)
    monkeypatch.setattr(sys, 'argv', [
        'cnmf', 'consensus', '--output-dir', str(tmp_path), '--name', 'streamed',
        '-k', '2', '--local-density-threshold', '2.0',
        '--local-neighborhood-size', '.5', '--streaming'])
    main()
    assert calls == [False, True, False]
    assert qc_flags == [False, True]
    for key in ['consensus_spectra', 'consensus_usages', 'gene_spectra_tpm',
                'gene_spectra_score', 'starcat_spectra']:
        a, b = [load_df_from_npz(o.paths[key] % (2, '2_0')) for o in objects]
        assert a.index.equals(b.index) and a.columns.equals(b.columns)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)

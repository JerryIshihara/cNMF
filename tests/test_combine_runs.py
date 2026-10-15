"""Synthetic saved spectra exercise pooling without expensive NMF fits."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from cnmf.cnmf import cNMF, load_df_from_npz, save_df_to_npz


def make_run(tmp_path, name, n=100, ks=(2, 3)):
    obj = cNMF(str(tmp_path), name)
    rows = []
    for k in ks:
        for i in range(n):
            rows.append(dict(n_components=k, iter=i, nmf_seed=42+i))
            h = pd.DataFrame(np.full((k, 4), i+1, dtype=np.float64),
                             columns=list("abcd"))
            save_df_to_npz(h, obj.paths["iter_spectra"] % (k, i))
    obj.save_nmf_iter_params(pd.DataFrame(rows), dict(solver="cd", beta_loss="frobenius"))
    return obj


def test_three_hundred(tmp_path, monkeypatch):
    sources = [make_run(tmp_path, "sample%d" % i) for i in range(3)]
    obj = cNMF(str(tmp_path), "pooled")
    with pytest.warns(UserWarning, match="expression inputs"):
        assert obj.combine(source_runs=[tmp_path / s.name for s in sources]) is None
    records = pd.read_csv(tmp_path / 'pooled/cnmf_tmp/pooled_replicates.tsv', sep='\t')
    for k in (2, 3):
        merged = load_df_from_npz(obj.paths["merged_spectra"] % k)
        assert merged.shape == (300*k, 4)
        assert merged.index.is_unique
        assert records[records.k == k].global_iter.tolist() == list(range(300))
        assert records[records.k == k].nmf_seed.tolist() == list(range(42,142))*3
    assert not Path(obj.paths["normalized_counts"]).exists()
    assert not Path(obj.paths["tpm"]).exists()
    original = cNMF.combine_nmf
    calls = []
    def record_call(self, k, **kwargs):
        calls.append(k)
        return original(self, k, **kwargs)
    monkeypatch.setattr(cNMF, 'combine_nmf', record_call)
    with pytest.warns(UserWarning):
        obj.combine(source_runs=[tmp_path / s.name for s in sources])
    assert calls == [2, 3]
    assert len(list((tmp_path / 'pooled').rglob('*.iter_*.df.npz'))) == 600


@pytest.mark.parametrize("error", ["genes", "dtype", "nan", "missing", "settings", "duplicate"])
def test_invalid_sources_publish_nothing(tmp_path, error):
    a, b = [make_run(tmp_path, name, n=2, ks=(2,)) for name in ("a", "b")]
    path = b.paths["iter_spectra"] % (2, 0)
    h = load_df_from_npz(path)
    if error == "genes":
        h = h[h.columns[::-1]]
    elif error == "dtype":
        h = h.astype(np.float32)
    elif error == "nan":
        h.iloc[0, 0] = np.nan
    save_df_to_npz(h, path)
    if error == "missing":
        Path(path).unlink()
    elif error == "settings":
        Path(b.paths["nmf_run_parameters"]).write_text("solver: mu\n")
    sources = [tmp_path / a.name, tmp_path / b.name]
    if error == "duplicate":
        sources[1] = sources[0]
    obj = cNMF(str(tmp_path), "out")
    with pytest.raises((ValueError, FileNotFoundError)):
        obj.combine(source_runs=sources)
    assert not list((tmp_path / "out").rglob("*.merged.df.npz"))
    assert not list((tmp_path / "out").glob(".combine-*"))


def test_k_filter(tmp_path):
    a = make_run(tmp_path, "a", n=2)
    obj = cNMF(str(tmp_path), "out")
    with pytest.warns(UserWarning):
        obj.combine(source_runs=[tmp_path / "a"], components=3)
    assert not Path(obj.paths["merged_spectra"] % 2).exists()
    assert load_df_from_npz(obj.paths["merged_spectra"] % 3).shape == (6, 4)


def test_legacy_combine(tmp_path):
    obj = make_run(tmp_path, "original", n=2, ks=(2,))
    obj.combine()
    assert load_df_from_npz(obj.paths["merged_spectra"] % 2).shape == (4, 4)


def test_missing_skip_provenance(tmp_path):
    a = make_run(tmp_path, "a", n=3, ks=(2,))
    Path(a.paths["iter_spectra"] % (2, 1)).unlink()
    obj = cNMF(str(tmp_path), "out")
    with pytest.warns(UserWarning):
        obj.combine(source_runs=[tmp_path / "a"], skip_missing_files=True)
    result = pd.read_csv(tmp_path / 'out/cnmf_tmp/pooled_replicates.tsv', sep='\t')
    assert result.local_iter.tolist() == [0, 1, 2]
    assert result.global_iter.tolist() == [0, 1, 2]
    merged = load_df_from_npz(obj.paths['merged_spectra'] % 2)
    assert merged.index.tolist() == ['iter0_topic1', 'iter0_topic2', 'iter2_topic1', 'iter2_topic2']


def test_cli(tmp_path, monkeypatch):
    import sys
    from cnmf.cnmf import main
    make_run(tmp_path, "a", n=2, ks=(2,))
    make_run(tmp_path, "b", n=2, ks=(2,))
    monkeypatch.setattr(sys, "argv", [
        "cnmf", "combine", "--output-dir", str(tmp_path), "--name", "pooled",
        "--source-runs", str(tmp_path / "a"), str(tmp_path / "b"),
        "-k", "2"])
    with pytest.warns(UserWarning):
        main()
    obj = cNMF(str(tmp_path), "pooled")
    assert load_df_from_npz(obj.paths["merged_spectra"] % 2).shape == (8, 4)


def test_interrupted_combine_regenerates(tmp_path, monkeypatch):
    make_run(tmp_path, 'a', n=2)
    obj = cNMF(str(tmp_path), 'out')
    original = cNMF.combine_nmf
    def interrupted(self, k, **kwargs):
        if k == 3:
            raise RuntimeError('interrupted')
        return original(self, k, **kwargs)
    monkeypatch.setattr(cNMF, 'combine_nmf', interrupted)
    with pytest.raises(RuntimeError):
        obj.combine(source_runs=[tmp_path / 'a'])
    calls = []
    def resumed(self, k, **kwargs):
        calls.append(k)
        return original(self, k, **kwargs)
    monkeypatch.setattr(cNMF, 'combine_nmf', resumed)
    with pytest.warns(UserWarning):
        obj.combine(source_runs=[tmp_path / 'a'])
    assert calls == [2, 3]
    monkeypatch.setattr(cNMF, 'combine_nmf', original)
    obj.combine(components=2)
    assert load_df_from_npz(obj.paths['merged_spectra'] % 2).shape == (4, 4)


def test_changed_source_regenerated(tmp_path):
    a = make_run(tmp_path, 'a', n=2, ks=(2,))
    obj = cNMF(str(tmp_path), 'out')
    with pytest.warns(UserWarning):
        obj.combine(source_runs=[tmp_path / 'a'])
    path = a.paths['iter_spectra'] % (2, 0)
    h = load_df_from_npz(path)
    save_df_to_npz(h * 2, path)
    cache = Path(obj.paths['local_density_cache'] % 2)
    cache.write_text('stale')
    unrelated = tmp_path / 'out' / 'notes.txt'
    unrelated.write_text('keep')
    with pytest.warns(UserWarning):
        obj.combine(source_runs=[tmp_path / 'a'])
    np.testing.assert_array_equal(load_df_from_npz(obj.paths['merged_spectra'] % 2).values[:2], h.values * 2)
    assert not cache.exists()
    assert unrelated.read_text() == 'keep'
    with pytest.warns(UserWarning):
        obj.combine(source_runs=[tmp_path / 'a'], components=2)
    assert len(list((tmp_path / 'out').rglob('*.iter_*.df.npz'))) == 2

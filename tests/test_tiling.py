from types import SimpleNamespace

import numpy as np
import pytest

from cnmf.gpunmf import utils


_GIB = 1 << 30


def fake_torch(*, free_bytes=6 * _GIB, total_bytes=8 * _GIB):
    calls = []
    cuda = SimpleNamespace(
        mem_get_info=lambda device: calls.append(device)
        or (free_bytes, total_bytes),
    )
    return SimpleNamespace(cuda=cuda), calls


def test_resolve_row_batch_uses_all_rows_when_cpu_is_automatic():
    torch, calls = fake_torch()
    X = np.zeros((100, 20), dtype=np.float32)

    row_batch, ratio = utils.resolve_row_batch(torch, "cpu", X, 2, 5)

    assert (row_batch, ratio) == (100, 1.0)
    assert calls == []


def test_resolve_row_batch_calculates_one_fitting_ratio_from_cuda_vram():
    torch, calls = fake_torch(free_bytes=5_000, total_bytes=20_000)
    X = np.zeros((100, 20), dtype=np.float32)

    row_batch, ratio = utils.resolve_row_batch(
        torch,
        "cuda:1",
        X,
        2,
        5,
        reserve_bytes=0,
        reserve_fraction=0,
    )

    assert calls == ["cuda:1"]
    assert ratio == 0.25
    assert row_batch == 25


def test_resolve_row_batch_uses_configured_ratio_without_vram_lookup():
    torch, calls = fake_torch()
    X = np.zeros((101, 20), dtype=np.float64)

    row_batch, ratio = utils.resolve_row_batch(
        torch,
        "cuda",
        X,
        2,
        5,
        configured_ratio=0.5,
    )

    assert (row_batch, ratio) == (50, 0.5)
    assert calls == []


@pytest.mark.parametrize("ratio", [0, -0.1, 1.1, np.nan, np.inf, "bad"])
def test_resolve_row_batch_rejects_invalid_configured_ratio(ratio):
    torch, _calls = fake_torch()
    X = np.zeros((10, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="row tiling ratio"):
        utils.resolve_row_batch(
            torch,
            "cpu",
            X,
            1,
            2,
            configured_ratio=ratio,
        )


def test_resolve_row_batch_rejects_input_without_rows():
    torch, _calls = fake_torch()
    X = np.zeros((0, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="at least one row"):
        utils.resolve_row_batch(torch, "cpu", X, 1, 2)

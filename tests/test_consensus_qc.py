import numpy as np

from cnmf.qc import edist


def test_edistance_matches_squared_distance_definition():
    spectra = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 2.0],
        [0.0, 3.0],
    ])
    grouping = np.array([1, 1, 2, 2])

    actual = edist(spectra, grouping)

    np.testing.assert_allclose(
        actual.to_numpy(),
        np.array([[0.0, 12.0], [12.0, 0.0]]),
        rtol=0,
        atol=1e-12,
    )
    assert actual.index.tolist() == [1, 2]
    assert actual.columns.tolist() == [1, 2]

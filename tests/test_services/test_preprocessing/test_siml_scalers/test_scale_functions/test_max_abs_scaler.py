import numpy as np
import scipy.sparse as sp

from siml.preprocessing.siml_scalers import SimlScalerWrapper


def test_max_abs_scaler():
    x = np.array([
        [1., 10., 100.],
        [2., -20., 200.],
        [3., 30., -300.],
    ])
    max_abs_scaler = SimlScalerWrapper("max_abs")
    max_abs_scaler.partial_fit(x)
    np.testing.assert_array_almost_equal(
        max_abs_scaler.converter.max_,
        [3., 30., 300.]
    )

    transformed_x = np.array([
        [1. / 3., 1. / 3., 1. / 3.],
        [2. / 3., -2. / 3., 2. / 3.],
        [3. / 3., 3. / 3., -3. / 3.],
    ])
    np.testing.assert_array_almost_equal(
        max_abs_scaler.transform(x),
        transformed_x
    )
    np.testing.assert_array_almost_equal(
        max_abs_scaler.inverse_transform(max_abs_scaler.transform(x)),
        x
    )


def test_max_abs_scaler_sparse():
    x = sp.coo_matrix(np.array([
        [1.],
        [0.],
        [3.],
        [-4.],
        [0.],
    ]))
    max_abs_scaler = SimlScalerWrapper("max_abs")
    max_abs_scaler.partial_fit(x)
    np.testing.assert_array_almost_equal(
        max_abs_scaler.converter.max_, [4.])

    transformed_x = np.array([
        [1. / 4.],
        [0. / 4.],
        [3. / 4.],
        [-4. / 4.],
        [0. / 4.],
    ])

    result = max_abs_scaler.transform(x)
    assert isinstance(result, sp.coo_matrix)

    np.testing.assert_array_almost_equal(
        result.toarray(), transformed_x
    )
    np.testing.assert_array_almost_equal(
        max_abs_scaler.inverse_transform(
            max_abs_scaler.transform(x)
        ).toarray(),
        x.toarray()
    )

import scipy.sparse as sp
import numpy as np

from siml.preprocessing.siml_scalers import SimlScalerWrapper


def test__sparse_standard_scaler():
    x1 = sp.coo_matrix(np.array([
        [1.],
        [0.],
        [3.],
        [-4.],
        [0.],
        [1.],
        [0.],
        [3.],
        [-4.],
        [0.],
    ]))
    x2 = sp.coo_matrix(np.array([
        [0.],
        [1.],
        [.3],
        [0.],
        [0.],
        [1.],
        [.3],
        [0.],
    ]))
    answer_var = np.mean(np.concatenate([x1.toarray(), x2.toarray()])**2)
    sp_std_scaler = SimlScalerWrapper("sparse_std")
    sp_std_scaler.partial_fit(x1)
    sp_std_scaler.partial_fit(x2)

    np.testing.assert_array_almost_equal(
        sp_std_scaler.converter.var_, answer_var)

    result = sp_std_scaler.transform(x1)
    assert isinstance(result, sp.coo_matrix)

    np.testing.assert_array_almost_equal(
        result.toarray(),
        x1.toarray() / answer_var**.5
    )
    np.testing.assert_array_almost_equal(
        sp_std_scaler.inverse_transform(
            sp_std_scaler.transform(x1)
        ).toarray(),
        x1.toarray()
    )


def test_sparse_standard_scaler_other_components():
    x1 = sp.coo_matrix(np.array([
        [1.],
        [0.],
        [3.],
        [-4.],
        [0.],
        [1.],
        [0.],
        [3.],
        [-4.],
        [0.],
    ]))
    x2 = sp.coo_matrix(np.array([
        [0.],
        [1.],
        [.3],
        [0.],
        [0.],
        [1.],
        [.3],
        [0.],
    ]))

    y1 = sp.coo_matrix(np.array([
        [1.],
        [0.],
        [13.],
        [-4.],
        [0.],
        [11.],
        [0.],
        [13.],
        [-14.],
        [0.],
    ]))
    y2 = sp.coo_matrix(np.array([
        [0.],
        [11.],
        [1.3],
        [0.],
        [0.],
        [11.],
        [1.3],
        [0.],
    ]))

    z1 = sp.coo_matrix(np.array([
        [21.],
        [0.],
        [3.],
        [-24.],
        [0.],
        [21.],
        [0.],
        [23.],
        [-24.],
        [0.],
    ]))
    z2 = sp.coo_matrix(np.array([
        [0.],
        [21.],
        [.3],
        [0.],
        [0.],
        [21.],
        [.3],
        [0.],
    ]))

    answer_var = np.mean(
        np.concatenate([x1.toarray(), x2.toarray()])**2
        + np.concatenate([y1.toarray(), y2.toarray()])**2
        + np.concatenate([z1.toarray(), z2.toarray()])**2
    )
    sp_std_scaler = SimlScalerWrapper(
        "sparse_std",
        other_components=['y', 'z']
    )
    sp_std_scaler.partial_fit(x1)
    sp_std_scaler.partial_fit(x2)
    sp_std_scaler.partial_fit(y1)
    sp_std_scaler.partial_fit(y2)
    sp_std_scaler.partial_fit(z1)
    sp_std_scaler.partial_fit(z2)

    np.testing.assert_array_almost_equal(
        sp_std_scaler.converter.var_, answer_var
    )

    result = sp_std_scaler.transform(x1)
    assert isinstance(result, sp.coo_matrix)

    np.testing.assert_array_almost_equal(
        sp_std_scaler.transform(x1).toarray(),
        x1.toarray() / answer_var**.5
    )
    np.testing.assert_array_almost_equal(
        sp_std_scaler.inverse_transform(
            sp_std_scaler.transform(x1)
        ).toarray(),
        x1.toarray()
    )

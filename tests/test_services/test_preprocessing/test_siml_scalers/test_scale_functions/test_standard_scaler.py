import numpy as np

from siml.preprocessing.siml_scalers.scale_functions import StandardScaler


def test__std_scale_values():
    n_element = int(1e5)
    epsilon = 1e-5

    interim_std_scale = np.random.rand(n_element, 3)

    scaler = StandardScaler(with_mean=False)
    preprocessed_data = scaler.fit_transform(interim_std_scale)

    np.testing.assert_almost_equal(
        interim_std_scale / (np.std(interim_std_scale, axis=0) + epsilon),
        preprocessed_data,
        decimal=3
    )

    # After transformed, std is almost 1
    np.testing.assert_almost_equal(
        np.std(preprocessed_data),
        1. + epsilon,
        decimal=3
    )


def test__standardize_values():
    n_element = int(1e5)
    epsilon = 1e-5

    interim_value = np.random.randn(n_element, 5) * 2 \
        + np.array([[.1, .2, .3, .4, .5]])

    scaler = StandardScaler()
    preprocessed = scaler.fit_transform(interim_value)

    np.testing.assert_almost_equal(
        (interim_value - np.mean(interim_value, axis=0))
        / (np.std(interim_value, axis=0) + epsilon),
        preprocessed,
        decimal=3
    )

    # After transformed, std is almost 1 and mean is almost zero
    np.testing.assert_almost_equal(
        np.std(preprocessed, axis=0),
        1. + epsilon,
        decimal=3
    )
    np.testing.assert_almost_equal(
        np.mean(preprocessed, axis=0),
        np.zeros(5),
        decimal=3
    )

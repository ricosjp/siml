import numpy as np

from siml.preprocessing.siml_scalers.scale_functions import IdentityScaler


def test__same_transform():
    scaler = IdentityScaler()

    n_element = int(1e5)
    input_data = np.random.randint(2, size=(n_element, 1))

    transformed_data = scaler.transform(input_data)

    np.testing.assert_array_almost_equal(
        input_data,
        transformed_data
    )


def test__use_diagonal():
    scaler = IdentityScaler()
    assert not scaler.use_diagonal

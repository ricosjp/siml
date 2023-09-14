import numpy as np
import pytest
import scipy.sparse as sp

from siml.preprocessing.siml_scalers import SimlScalerWrapper
from siml.preprocessing.siml_scalers import scale_functions


@pytest.fixture
def scalers_name() -> str:
    scalers_name = [
        "identity", "standardize", "std_scale", "sparse_std",
        "isoam_scale", "min_max", "max_abs"
    ]
    return scalers_name


@pytest.fixture
def sample_data() -> np.ndarray:
    return np.random.rand(100, 3)


@pytest.fixture
def sample_sparse_data() -> sp.csc_matrix:
    return sp.csc_matrix(np.random.rand(100, 3))


@pytest.mark.parametrize("scaler_name, cls", [
    ("identity", scale_functions.IdentityScaler),
    ("standardize", scale_functions.StandardScaler),
    ("std_scale", scale_functions.StandardScaler),
    ("sparse_std", scale_functions.SparseStandardScaler),
    ("isoam_scale", scale_functions.IsoAMScaler),
    ("min_max", scale_functions.MinMaxScaler),
    ("max_abs", scale_functions.MaxAbsScaler)
])
def test__initialized_class(scaler_name, cls):
    kwards = {}
    if scaler_name == "isoam_scale":
        kwards = {"other_components": ["a", "b"]}
    wrapper = SimlScalerWrapper(scaler_name, **kwards)
    assert isinstance(wrapper.converter, cls)


def test__can_initialize(scalers_name):
    for name in scalers_name:
        if name == "isoam_scale":
            continue

        _ = SimlScalerWrapper(name)


def test__can_not_initialize_isoam_with_default():
    with pytest.raises(ValueError):
        _ = SimlScalerWrapper("isoam_scale")


@pytest.mark.parametrize("name", [
    "std_scale"
])
def test__with_mean_false(name):
    scaler = SimlScalerWrapper(name)
    assert not scaler.converter.with_mean


def test__transform(sample_data):
    scaler = SimlScalerWrapper("standardize")
    scaler.partial_fit(sample_data)
    transformed_data = scaler.transform(sample_data)

    assert transformed_data.shape == sample_data.shape

    means = np.mean(transformed_data, axis=0)
    np.testing.assert_array_almost_equal(
        means,
        np.zeros(means.shape),
        decimal=3
    )


def test__transform_sparse_data(sample_sparse_data):
    scaler = SimlScalerWrapper("std_scale")
    scaler.partial_fit(sample_sparse_data)
    transformed_data = scaler.transform(sample_sparse_data)

    assert transformed_data.shape == sample_sparse_data.shape
    assert isinstance(transformed_data, sp.coo_matrix)


def test__dumped_dict(sample_data):
    scaler = SimlScalerWrapper("std_scale")
    scaler.partial_fit(sample_data)

    dumped_data = scaler.get_dumped_dict()
    assert dumped_data["preprocess_converter"]["n_samples_seen_"] \
        == len(sample_data)
    assert dumped_data["method"] == "std_scale"


def test__dump_and_load(sample_data):
    scaler = SimlScalerWrapper("std_scale")
    scaler.partial_fit(sample_data)

    dumped_data = scaler.get_dumped_dict()
    new_scaler = SimlScalerWrapper.create(dumped_data)

    assert vars(scaler.converter) == vars(new_scaler.converter)

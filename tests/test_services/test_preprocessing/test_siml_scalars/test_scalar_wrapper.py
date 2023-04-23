import pytest

from siml.preprocessing.siml_scalers import SimlScalerWrapper


@pytest.fixture
def scalers_name() -> str:
    scalers_name = [
        "identity", "standardize", "std_scale", "sparse_std",
        "isoam_scale", "min_max", "max_abs"
    ]
    return scalers_name


def test__can_initialize(scalers_name):
    for name in scalers_name:
        if name == "isoam_scale":
            continue

        _ = SimlScalerWrapper(name)


def test__can_not_initialize_isoam_with_default():
    with pytest.raises(ValueError):
        _ = SimlScalerWrapper("isoam_scale")

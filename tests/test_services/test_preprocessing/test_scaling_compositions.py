import pathlib

import pytest

from siml import setting
from siml.preprocessing import ScalersComposition


@pytest.fixture
def scalers_composition():
    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/deform/data.yml')
    )
    composite = ScalersComposition(
        preprocess_dict=main_setting.preprocess
    )
    return composite


@pytest.mark.parametrize("arg1, arg2", [
    ("tensor_gauss_strain1", "tensor_gauss_strain2"),
    ("tensor_gauss_strain1", "tensor_gauss_strain3"),
    ("tensor_gauss_strain1", "tensor_gauss_strain4"),
    ("x_grad", "y_grad"),
    ("x_grad", "z_grad"),
    ("x_grad", "y_grad_2"),
    ("x_grad", "z_grad_2"),
])
def test__is_same_scalers(scalers_composition, arg1, arg2):
    resolver: ScalersComposition = scalers_composition
    scaler_1 = resolver.get_scaler(arg1)
    scaler_2 = resolver.get_scaler(arg2)
    assert id(scaler_1) == id(scaler_2)


@pytest.mark.parametrize("arg1, arg2", [
    ("tensor_gauss_strain1", "tensor_strain"),
    ("x_grad", "x_grad_2")
])
def test__is_not_same_scalers(scalers_composition, arg1, arg2):
    resolver: ScalersComposition = scalers_composition
    scaler_1 = resolver.get_scaler(arg1)
    scaler_2 = resolver.get_scaler(arg2)
    assert id(scaler_1) != id(scaler_2)

import pathlib
import pickle

import numpy as np
import pytest

from siml import setting
from siml.path_like_objects import SimlFileBuilder
from siml.preprocessing import ScalersComposition


@pytest.fixture
def scalers_composition():
    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/deform/data.yml')
    )
    composite = ScalersComposition.create_from_dict(
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


def test__is_same_after_load_dumped_data(scalers_composition):
    composition: ScalersComposition = scalers_composition

    sample_data = {
        "tensor_stress": [
            SimlFileBuilder.numpy_file(
                pathlib.Path(
                    "tests/data/deform/interim/train/"
                    "tet2_3_modulusx0.9000/tensor_stress.npy"
                )
            )
        ]
    }
    composition.lazy_partial_fit(sample_data)
    items_1 = composition.get_scaler("tensor_stress").get_dumped_dict()

    dumped_dict = composition.get_dumped_object()
    assert dumped_dict[
        "tensor_stress"
    ]["preprocess_converter"]["n_samples_seen_"] > 0

    dumped_dict.pop("variable_name_to_scalers")
    scalers_dict = composition._load_scalers(dumped_dict)

    items_2 = scalers_dict["tensor_stress"].get_dumped_dict()
    assert items_1 == items_2


def test__load_converters_pkl():
    preprocessors_file = pathlib.Path('tests/data/prepost/preprocessors.pkl')
    real_file_converter = ScalersComposition.create_from_file(
        preprocessors_file
    )

    with open(preprocessors_file, "rb") as fr:
        dict_data = pickle.load(fr)

    np.testing.assert_almost_equal(
        real_file_converter.get_scaler('standardize').converter.var_,
        dict_data['standardize']['preprocess_converter']['var_']
    )


@pytest.mark.parametrize("pkl_path", [
    pathlib.Path("tests/data/old_pkl_files/preprocessors.pkl")
])
def test__can_load_old_format_pkl(pkl_path):
    _ = ScalersComposition.create_from_file(
        pkl_path
    )

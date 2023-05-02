import pathlib
import shutil
import pickle

import numpy as np
import pytest
import scipy.sparse as sp

from siml import setting
from siml.preprocessing.converter import RawConverter
from siml.preprocessing.scaling_converter import (
    PreprocessInnerSettings,
    ScalingConverter
)

import preprocess


# region  Test for PreprocessInnerSetting

@pytest.fixture
def inner_setting():
    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/heat_boundary/data.yml')
    )
    inner_setting = PreprocessInnerSettings(
        preprocess_dict=main_setting.preprocess,
        interim_directories=main_setting.data.interim,
        preprocessed_root=main_setting.data.preprocessed_root
    )
    return inner_setting


def test__collect_scaler_fitting_files(inner_setting):
    inner_setting: PreprocessInnerSettings = inner_setting
    siml_files = inner_setting.get_scaler_fitting_files("node")

    assert len(siml_files) > 0
    for siml_file in siml_files:
        assert str(siml_file.file_path.name).startswith("node")


# endregion

# region Test for ScalingConverter

def setup_sample_data_setting():
    data_setting = setting.DataSetting(
        interim=pathlib.Path('tests/data/prepost/interim'),
        preprocessed=pathlib.Path('tests/data/prepost/preprocessed'),
        pad=False
    )
    return data_setting


def setup_sample_data_name():
    return ['a', 'b']


@pytest.fixture(scope="module")
def prepare_sample_dataset():
    data_setting: setting.DataSetting = setup_sample_data_setting()
    preprocess_setting = setting.PreprocessSetting(
        {
            'identity': 'identity',
            'std_scale': 'std_scale',
            'standardize': 'standardize'
        }
    )
    main_setting = setting.MainSetting(
        preprocess=preprocess_setting.preprocess,
        data=data_setting,
        replace_preprocessed=False
    )
    main_setting.preprocess['std_scale']['componentwise'] = True
    main_setting.preprocess['standardize']['componentwise'] = True

    # Clean up data
    shutil.rmtree(data_setting.interim_root, ignore_errors=True)
    shutil.rmtree(data_setting.preprocessed_root, ignore_errors=True)
    data_setting.preprocessed_root.mkdir(parents=True)

    # Create data
    interim_paths = [
        data_setting.interim_root / name
        for name in setup_sample_data_name()
    ]
    for i, interim_path in enumerate(interim_paths):
        interim_path.mkdir(parents=True)
        n_element = int(1e5)
        identity = np.random.randint(2, size=(n_element, 1))
        std_scale = np.random.rand(n_element, 3) * 5 * i
        standardize = np.random.randn(n_element, 5) * 2 * i \
            + i * np.array([[.1, .2, .3, .4, .5]])
        np.save(interim_path / 'identity.npy', identity)
        np.save(interim_path / 'std_scale.npy', std_scale)
        np.save(interim_path / 'standardize.npy', standardize)
        (interim_path / 'converted').touch()

    # Preprocess data
    preprocessor = ScalingConverter(main_setting)
    preprocessor.fit_transform()


def test_preprocessor_sample_dataset(prepare_sample_dataset):
    # HACK: This test will be deprecated in the future
    # This should be divided into more small unit tests
    # For example,
    # test for each scaler should be written in "test_scale_functions"
    data_setting: setting.DataSetting = \
        setup_sample_data_setting()
    interim_paths = [
        data_setting.interim_root / name
        for name in setup_sample_data_name()
    ]

    # Test preprocessed data is as desired
    epsilon = 1e-5
    preprocessed_paths = [
        data_setting.preprocessed_root / name
        for name in setup_sample_data_name()
    ]

    int_identity = np.concatenate([
        np.load(p / 'identity.npy') for p in interim_paths])
    pre_identity = np.concatenate([
        np.load(p / 'identity.npy') for p in preprocessed_paths])

    np.testing.assert_almost_equal(
        int_identity, pre_identity, decimal=3)

    int_std_scale = np.concatenate([
        np.load(p / 'std_scale.npy') for p in interim_paths])
    pre_std_scale = np.concatenate([
        np.load(p / 'std_scale.npy') for p in preprocessed_paths])

    np.testing.assert_almost_equal(
        int_std_scale / (np.std(int_std_scale, axis=0) + epsilon),
        pre_std_scale, decimal=3)
    np.testing.assert_almost_equal(
        np.std(pre_std_scale), 1. + epsilon, decimal=3)

    int_standardize = np.concatenate([
        np.load(p / 'standardize.npy') for p in interim_paths])
    pre_standardize = np.concatenate([
        np.load(p / 'standardize.npy') for p in preprocessed_paths])

    np.testing.assert_almost_equal(
        (int_standardize - np.mean(int_standardize, axis=0))
        / (np.std(int_standardize, axis=0) + epsilon),
        pre_standardize,
        decimal=3
    )
    np.testing.assert_almost_equal(
        np.std(pre_standardize, axis=0),
        1. + epsilon,
        decimal=3
    )
    np.testing.assert_almost_equal(
        np.mean(pre_standardize, axis=0),
        np.zeros(5),
        decimal=3
    )


def test_preprocess_deform():
    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/deform/data.yml')
    )
    main_setting.data.interim = [
        pathlib.Path('tests/data/deform/test_prepost/interim')
    ]
    main_setting.data.preprocessed = [
        pathlib.Path('tests/data/deform/test_prepost/preprocessed')
    ]

    shutil.rmtree(main_setting.data.interim_root, ignore_errors=True)
    shutil.rmtree(main_setting.data.preprocessed_root, ignore_errors=True)

    raw_converter = RawConverter(
        main_setting,
        conversion_function=preprocess.ConversionFunction())
    raw_converter.convert()

    preprocessor = ScalingConverter(main_setting)
    preprocessor.fit_transform()

    interim_strain = np.load(
        'tests/data/deform/test_prepost/interim/train/'
        'tet2_3_modulusx1.0000/elemental_strain.npy')
    preprocessed_strain = np.load(
        'tests/data/deform/test_prepost/preprocessed/train/'
        'tet2_3_modulusx1.0000/elemental_strain.npy')
    ratio_strain = interim_strain / preprocessed_strain
    np.testing.assert_almost_equal(
        ratio_strain - np.mean(ratio_strain), 0.)

    interim_y_grad = sp.load_npz(
        'tests/data/deform/test_prepost/interim/train/'
        'tet2_3_modulusx1.0000/y_grad.npz')
    preprocessed_y_grad = sp.load_npz(
        'tests/data/deform/test_prepost/preprocessed/train/'
        'tet2_3_modulusx1.0000/y_grad.npz')

    ratio_y_grad = interim_y_grad.data \
        / preprocessed_y_grad.data
    np.testing.assert_almost_equal(np.var(ratio_y_grad), 0.)


def test__time_series_initial_state_in_ode_data():
    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/ode/data.yml'))

    shutil.rmtree(main_setting.data.preprocessed_root, ignore_errors=True)
    preprocessor = ScalingConverter(main_setting, force_renew=True)
    preprocessor.fit_transform()
    data_directory = main_setting.data.preprocessed_root / 'train/0'
    y0 = np.load(data_directory / 'y0.npy')
    y0_initial = np.load(data_directory / 'y0_initial.npy')
    np.testing.assert_almost_equal(y0[0], y0_initial[0])
    np.testing.assert_almost_equal(y0_initial - y0_initial[0], 0.)


def test_preprocess_power():
    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/deform/power.yml')
    )

    shutil.rmtree(main_setting.data.preprocessed_root, ignore_errors=True)

    preprocessor = ScalingConverter(main_setting, force_renew=True)
    preprocessor.fit_transform()

    data_directory = main_setting.data.preprocessed_root \
        / 'train/tet2_3_modulusx0.9000'
    preprocessed_x_grad = sp.load_npz(data_directory / 'x_grad.npz')
    reference_x_grad = sp.load_npz(
        'tests/data/deform/interim/train/tet2_3_modulusx0.9000'
        '/x_grad.npz').toarray()
    with open(
        main_setting.data.preprocessed_root / 'preprocessors.pkl',
        'rb'
    ) as f:
        preprocess_converter_setting = pickle.load(f)['x_grad'][
            'preprocess_converter']

    std = preprocess_converter_setting['std_']

    scaler = preprocessor._scalers.get_scaler('x_grad')
    scaler.converter.power = 0.5
    np.testing.assert_almost_equal(
        preprocessed_x_grad.toarray() * std**.5,
        reference_x_grad
    )

    np.testing.assert_almost_equal(
        scaler.inverse_transform(preprocessed_x_grad).toarray(),
        reference_x_grad
    )


def test_preprocess_interim_list():
    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/list/data.yml')
    )
    shutil.rmtree(main_setting.data.preprocessed_root, ignore_errors=True)

    preprocessor = ScalingConverter(main_setting)
    preprocessor.fit_transform()

    assert pathlib.Path(
        'tests/data/list/preprocessed/data/tet2_3_modulusx0.9500'
    ).exists()

    assert pathlib.Path(
        'tests/data/list/preprocessed/data/tet2_4_modulusx0.9000'
    ).exists()

# endregion

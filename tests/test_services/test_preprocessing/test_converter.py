import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import femio
from siml import setting
from siml import util
from siml.preprocessing import converter
import preprocess


class LoadFunction(converter.ILoadFunction):
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        data_files: list[Path],
        raw_path: Path
    ) -> tuple[dict, femio.FEMData]:
        # To be used in test_convert_raw_data_bypass_femio
        df = pd.read_csv(data_files[0], header=0, index_col=None)
        return {
            'a': np.reshape(df['a'].to_numpy(), (-1, 1)),
            'b': np.reshape(df['b'].to_numpy(), (-1, 1)),
            'c': np.reshape(df['c'].to_numpy(), (-1, 1))}, None


class FluidLoadFunction(converter.ILoadFunction):
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        data_files: list[Path],
        raw_path: Path
    ) -> tuple[dict, femio.FEMData]:
        fem_data = femio.read_files('vtu', data_files)
        dict_data = {
            'u': fem_data.nodal_data.get_attribute_data('U'),
            'p': fem_data.nodal_data.get_attribute_data('p'),
        }
        return dict_data, fem_data


class FilterFunction(converter.IFilterFunction):
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        fem_data: femio.FEMData,
        raw_path: Path = None,
        dict_data: dict[str, np.ndarray] = None
    ) -> bool:
        # To be used in test_convert_raw_data_with_filter_function
        strain = fem_data.elemental_data.get_attribute_data('ElementalSTRAIN')
        return np.max(np.abs(strain)) < 1e2


def test_convert_raw_data_bypass_femio():
    data_setting = setting.DataSetting(
        raw=Path('tests/data/csv_prepost/raw'),
        interim=Path('tests/data/csv_prepost/interim'))
    conversion_setting = setting.ConversionSetting(
        required_file_names=['*.csv'], skip_femio=True)

    main_setting = setting.MainSetting(
        data=data_setting, conversion=conversion_setting)

    shutil.rmtree(data_setting.interim_root, ignore_errors=True)
    shutil.rmtree(data_setting.preprocessed_root, ignore_errors=True)

    rc = converter.RawConverter(
        main_setting, recursive=True, load_function=LoadFunction())
    rc.convert()

    interim_directory = data_setting.interim_root / 'train/1'
    expected_a = np.array([[1], [2], [3], [4]])
    expected_b = np.array([[2.1], [4.1], [6.1], [8.1]])
    expected_c = np.array([[3.2], [7.2], [8.2], [10.2]])
    np.testing.assert_almost_equal(
        np.load(interim_directory / 'a.npy'), expected_a)
    np.testing.assert_almost_equal(
        np.load(interim_directory / 'b.npy'), expected_b, decimal=5)
    np.testing.assert_almost_equal(
        np.load(interim_directory / 'c.npy'), expected_c, decimal=5)


def test__is_same_results_when_not_save_results():
    data_setting = setting.DataSetting(
        raw=Path('tests/data/csv_prepost/raw'),
        interim=Path('tests/data/csv_prepost/interim'))
    conversion_setting = setting.ConversionSetting(
        required_file_names=['*.csv'], skip_femio=True)

    main_setting = setting.MainSetting(
        data=data_setting, conversion=conversion_setting)

    shutil.rmtree(data_setting.interim_root, ignore_errors=True)
    shutil.rmtree(data_setting.preprocessed_root, ignore_errors=True)

    rc = converter.RawConverter(
        main_setting, recursive=True, load_function=LoadFunction())
    rc.convert()

    rc_2 = converter.RawConverter(
        main_setting, recursive=True,
        load_function=LoadFunction(), force_renew=True)
    results = rc_2.convert(return_results=True)

    case_names = ["0", "1", "2"]
    value_names = ["a", "b", "c"]
    for i, name in enumerate(case_names):
        interim_directory = data_setting.interim_root / f'train/{name}'
        raw_directory = Path('tests/data/csv_prepost/raw') / f"train/{name}"
        result = results[str(raw_directory)]

        for value_name in value_names:
            value_array = np.load(interim_directory / f'{value_name}.npy')
            np.testing.assert_array_almost_equal(
                value_array, result[0][value_name]
            )


def test_convert_raw_data_with_filter_function():
    main_setting = setting.MainSetting.read_settings_yaml(
        Path('tests/data/test_prepost_to_filter/data.yml'))
    shutil.rmtree(main_setting.data.interim_root, ignore_errors=True)

    raw_converter = converter.RawConverter(
        main_setting, filter_function=FilterFunction())
    raw_converter.convert()

    actual_directories = sorted(util.collect_data_directories(
        main_setting.data.interim,
        required_file_names=['elemental_strain.npy']))
    expected_directories = sorted([
        main_setting.data.interim_root / 'tet2_3_modulusx0.9000',
        main_setting.data.interim_root / 'tet2_3_modulusx1.1000',
        main_setting.data.interim_root / 'tet2_4_modulusx1.0000',
        main_setting.data.interim_root / 'tet2_4_modulusx1.1000'])
    np.testing.assert_array_equal(actual_directories, expected_directories)


def test_convert_heat_time_series():
    main_setting = setting.MainSetting.read_settings_yaml(
        Path('tests/data/heat_time_series/data.yml'))

    shutil.rmtree(main_setting.data.interim_root, ignore_errors=True)

    rc = converter.RawConverter(
        main_setting, recursive=True, write_ucd=False,
        conversion_function=preprocess.ConversionFunctionHeatTimeSeries()
    )
    rc.convert()


def test_save_dtype_is_applied():
    data_setting = setting.DataSetting(
        raw=Path('tests/data/csv_prepost/raw'),
        interim=Path('tests/data/csv_prepost/interim'))
    conversion_setting = setting.ConversionSetting(
        required_file_names=['*.csv'], skip_femio=True)

    main_setting = setting.MainSetting(
        data=data_setting, conversion=conversion_setting,
        misc={"save_dtype_dict": {"a": 'int32'}})

    shutil.rmtree(data_setting.interim_root, ignore_errors=True)
    shutil.rmtree(data_setting.preprocessed_root, ignore_errors=True)

    rc = converter.RawConverter(
        main_setting, recursive=True, load_function=LoadFunction())
    rc.convert()

    interim_directory = data_setting.interim_root / 'train/1'
    interim_data = np.load(interim_directory / 'a.npy')

    assert interim_data.dtype == np.int32


def test_convert_raw_data_with_load_function_and_additional_variables():
    main_setting = setting.MainSetting.read_settings_yaml(
        Path('tests/data/additional_variables/data.yml'))
    shutil.rmtree(main_setting.data.interim_root, ignore_errors=True)

    raw_converter = converter.RawConverter(
        main_setting, load_function=FluidLoadFunction())
    raw_converter.convert()

    actual_directory = sorted(util.collect_data_directories(
        main_setting.data.interim,
        required_file_names=['a.npy']))[0]
    answer_fem_data = femio.read_files(
        'vtu', 'tests/data/additional_variables/raw/step1.0_u1.0/mesh.vtu')

    actual_a = np.load(actual_directory / 'a.npy')
    np.testing.assert_almost_equal(
        actual_a, answer_fem_data.nodal_data.get_attribute_data('a'))

    actual_b = np.load(actual_directory / 'b.npy')
    np.testing.assert_almost_equal(
        actual_b, answer_fem_data.nodal_data.get_attribute_data('b'))

    actual_u = np.load(actual_directory / 'u.npy')
    np.testing.assert_almost_equal(
        actual_u, answer_fem_data.nodal_data.get_attribute_data('U'))

    actual_p = np.load(actual_directory / 'p.npy')
    np.testing.assert_almost_equal(
        actual_p, answer_fem_data.nodal_data.get_attribute_data('p'))


def test__create_save_function():
    main_setting = setting.MainSetting.read_settings_yaml(
        Path('tests/data/test_prepost_to_filter/data.yml'))

    def user_save_function(
        fem_data: femio.FEMData,
        dict_data: dict,
        output_directory: Path,
        force_renew: bool
    ):
        pass

    raw_converter = converter.RawConverter(
        main_setting,
        filter_function=FilterFunction(),
        save_function=user_save_function
    )

    save_function = raw_converter._create_save_function()
    assert save_function.user_save_function == user_save_function

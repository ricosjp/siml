import pytest
import pathlib

from siml import setting
from siml.preprocessing.scaling_converter import PreprocessInnerSettings
from siml.preprocessing import ScalingConverter


@pytest.fixture
def collect_inner_setting():
    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/heat_boundary/data.yml')
    )
    inner_setting = PreprocessInnerSettings(
        preprocess_dict=main_setting.preprocess,
        interim_directories=main_setting.data.interim,
        preprocessed_root=main_setting.data.preprocessed_root
    )
    return inner_setting


def test__collect_scaler_fitting_files(collect_inner_setting):
    inner_setting: PreprocessInnerSettings = collect_inner_setting
    dirs = inner_setting.collect_scaler_fitting_files("node")
    for d in dirs:
        print(d)
    assert len(dirs) > 0


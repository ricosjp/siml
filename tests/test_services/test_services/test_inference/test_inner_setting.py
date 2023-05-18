import pathlib
from unittest import mock
import pytest

from siml import setting
from siml.services.inference import InnerInfererSetting


def test__get_snapshot_file_path_set_by_main_setting():
    main_setting = setting.MainSetting()
    model_path = pathlib.Path(
        "tests/data/simplified/pretrained/snapshot_epoch_1000.pth"
    )
    main_setting.inferer.model = model_path

    inner_setting = InnerInfererSetting(
        main_setting=main_setting
    )
    assert inner_setting.get_snapshot_file_path() == model_path


@pytest.mark.parametrize("force_path,expect_path", [
    (
        "tests/data/simplified/pretrained/snapshot_epoch_1000.pth",
        "tests/data/simplified/pretrained/snapshot_epoch_1000.pth",
    )
])
def test__get_snapshot_file_path_set_by_explicit(force_path, expect_path):
    main_setting = setting.MainSetting()
    force_model_path = pathlib.Path(force_path)

    inner_setting = InnerInfererSetting(
        main_setting=main_setting,
        force_model_path=force_model_path
    )
    assert inner_setting.get_snapshot_file_path() == pathlib.Path(expect_path)


def test__get_snapshot_file_path_specified():
    main_setting = setting.MainSetting()
    main_setting.trainer.snapshot_choise_method = 'specified'
    force_model_path = pathlib.Path("tests/data/simplified/pretrained")

    inner_setting = InnerInfererSetting(
        main_setting=main_setting,
        force_model_path=force_model_path,
        infer_epoch=1000
    )
    expect_path = pathlib.Path(
        "tests/data/simplified/pretrained/snapshot_epoch_1000.pth"
    )
    assert inner_setting.get_snapshot_file_path() == expect_path


@pytest.mark.parametrize("data_directory", [
    "./tests/sample",
    "./tests/aaa/bbb"
    "./aaa"
])
def test__get_write_simulation_case_dir_when_perform_preprocess(
    data_directory
):
    main_setting = setting.MainSetting()
    main_setting.inferer.perform_preprocess = True
    inner_setting = InnerInfererSetting(
        main_setting=main_setting
    )
    actual = inner_setting.get_write_simulation_case_dir(data_directory) 
    assert actual == data_directory


@pytest.mark.parametrize("force_input_path", [
    "./tests/sample.pkl",
    "./aaa/sample.pkl"
])
def test__get_converter_parameters_pkl_path_force_input(force_input_path):
    main_setting = setting.MainSetting()
    force_input_path = pathlib.Path(force_input_path)
    inner_setting = InnerInfererSetting(
        main_setting=main_setting,
        force_converter_parameters_pkl=force_input_path
    )

    actual = inner_setting.get_converter_parameters_pkl_path()
    assert actual == force_input_path


@pytest.mark.parametrize("pkl_path, root_dir, expect", [
    (None, "./tests/sample", "./tests/sample/preprocessors.pkl"),
    ("./tests/sample/test.pkl", "./aaa", "./tests/sample/test.pkl")
])
def test__get_converter_parameters_pkl_path(pkl_path, root_dir, expect):
    main_setting = setting.MainSetting()
    if pkl_path is not None:
        main_setting.inferer.converter_parameters_pkl = pathlib.Path(pkl_path)

    with mock.patch(
        'siml.setting.DataSetting',
        new_callable=mock.PropertyMock
    ) as mock_setting:
        mock_setting.return_value = pathlib.Path(root_dir)

        inner_setting = InnerInfererSetting(
            main_setting=main_setting
        )

        actual = inner_setting.get_converter_parameters_pkl_path()
        assert actual == pathlib.Path(expect)

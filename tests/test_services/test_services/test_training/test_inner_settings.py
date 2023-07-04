from unittest import mock
import pathlib

import pytest

from siml.setting import MainSetting, TrainerSetting
from siml.path_like_objects import SimlDirectory, SimlFileBuilder
from siml.services.training import InnerTrainingSetting


@pytest.fixture
def setup_main_setting() -> MainSetting:
    # To pass root_validator in InnerTrainingSetting
    main_setting = MainSetting(
        trainer=TrainerSetting(
            inputs=[{"name": "x", "dim": 10}],
            outputs=[{"name": "y", "dim": 10}]
        )
    )
    return main_setting


def test__not_allowed_both_pretrain_and_restart(setup_main_setting):
    main_settings: MainSetting = setup_main_setting
    main_settings.trainer.restart_directory = pathlib.Path(
        "tests/data/somewhere"
    )
    main_settings.trainer.pretrain_directory = pathlib.Path(
        "tests/data/somewhere"
    )

    with pytest.raises(ValueError):
        _ = InnerTrainingSetting(main_setting=main_settings)


def test__load_restart_settings_content(setup_main_setting):
    main_settings: MainSetting = setup_main_setting
    main_settings.trainer.restart_directory = \
        pathlib.Path("tests/data/somewhere")
    with mock.patch.object(SimlDirectory, "find_yaml_file") as mocked:
        siml_file = SimlFileBuilder.yaml_file(
            pathlib.Path("tests/data/linear/linear.yml")
        )
        mocked.return_value = siml_file
        inner_setting = InnerTrainingSetting(main_setting=main_settings)

        assert inner_setting.trainer_setting.batch_size == 2
        assert inner_setting.trainer_setting.n_epoch == 3000

        assert inner_setting.main_setting.model.blocks[0].type\
            == "adjustable_mlp"


def test__inherit_values_when_restart(setup_main_setting):
    main_settings: MainSetting = setup_main_setting
    key = bytes(b'sample_test')
    output_directory = pathlib.Path("tests/data/sample/output")
    restart_directory = pathlib.Path("tests/data/somewhere")
    main_settings.data.encrypt_key = key
    main_settings.trainer.output_directory = output_directory
    main_settings.trainer.restart_directory = restart_directory

    with mock.patch.object(SimlDirectory, "find_yaml_file") as mocked:
        siml_file = SimlFileBuilder.yaml_file(
            pathlib.Path("tests/data/linear/linear.yml")
        )
        mocked.return_value = siml_file
        inner_setting = InnerTrainingSetting(main_setting=main_settings)

        settings = inner_setting.main_setting
        assert settings.get_crypt_key() == key
        assert settings.trainer.output_directory == output_directory
        assert settings.trainer.restart_directory == restart_directory

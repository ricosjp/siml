from unittest import mock
import pathlib

import pytest

from siml.setting import MainSetting
from siml.path_like_objects import SimlDirectory
from siml.services.training import InnerTrainingSetting


def test__not_allowed_both_pretrain_and_restart():
    main_settings = MainSetting()
    main_settings.trainer.restart_directory = pathlib.Path(
        "tests/data/somewhere"
    )
    main_settings.trainer.pretrain_directory = pathlib.Path(
        "tests/data/somewhere"
    )

    with pytest.raises(ValueError):
        _ = InnerTrainingSetting(main_settings=main_settings)


def test__load_restart_settings_content():
    main_settings = MainSetting()
    main_settings.trainer.restart_directory = \
        pathlib.Path("tests/data/somewhere")
    with mock.patch.object(SimlDirectory, "find_yaml_file") as mocked:
        mocked.return_value = pathlib.Path(
            "tests/data/linear/linear.yml"
        )
        inner_setting = InnerTrainingSetting(main_settings=main_settings)

        assert inner_setting.trainer_setting.batch_size == 2
        assert inner_setting.trainer_setting.n_epoch == 3000

        assert inner_setting.main_settings.model.blocks[0].type\
            == "adjustable_mlp"


def test__inherit_values_when_restart():
    main_settings = MainSetting()
    key = bytes(b'sample_test')
    output_directory = pathlib.Path("tests/data/sample/output")
    restart_directory = pathlib.Path("tests/data/somewhere")
    main_settings.data.encrypt_key = key
    main_settings.trainer.output_directory = output_directory
    main_settings.trainer.restart_directory = restart_directory

    with mock.patch.object(SimlDirectory, "find_yaml_file") as mocked:
        mocked.return_value = pathlib.Path(
            "tests/data/linear/linear.yml"
        )
        inner_setting = InnerTrainingSetting(main_settings=main_settings)

        settings = inner_setting.main_settings
        assert settings.get_crypt_key() == key
        assert settings.trainer.output_directory == output_directory
        assert settings.trainer.restart_directory == restart_directory

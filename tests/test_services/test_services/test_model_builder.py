import pytest
import pathlib

from siml import setting
from siml.path_like_objects import SimlFileBuilder
from siml.services import ModelBuilder, ModelEnvironmentSetting

TEST_DIR = pathlib.Path("tests/data/simplified/pretrained")


@pytest.fixture
def prepare_settings():
    yaml_path = TEST_DIR / "settings.yaml"
    main_setting = setting.MainSetting.read_settings_yaml(yaml_path)
    env_setting = ModelEnvironmentSetting(
        gpu_id=main_setting.trainer.gpu_id,
        seed=main_setting.trainer.seed,
        data_parallel=main_setting.trainer.data_parallel,
        model_parallel=main_setting.trainer.model_parallel,
        time_series=main_setting.trainer.time_series
    )
    return main_setting, env_setting


@pytest.mark.parametrize("flag", [
    True, False
])
def test__state_dict_strict(flag, prepare_settings):
    main_settings, env_setting = prepare_settings
    main_settings.trainer.state_dict_strict = flag

    builder = ModelBuilder(
        model_setting=main_settings.model,
        trainer_setting=main_settings.trainer,
        env_setting=env_setting
    )

    assert builder.state_dict_strict == flag


def test__create_initialized(prepare_settings):
    main_settings, env_setting = prepare_settings
    builder = ModelBuilder(
        model_setting=main_settings.model,
        trainer_setting=main_settings.trainer,
        env_setting=env_setting
    )

    model = builder.create_initialized()

    # HACK: If initialized, train mode is true.
    # Any other better tests ?
    assert model.training


def test__enable_create_loaded(prepare_settings):
    main_settings, env_setting = prepare_settings
    builder = ModelBuilder(
        model_setting=main_settings.model,
        trainer_setting=main_settings.trainer,
        env_setting=env_setting
    )
    checkpoint_file = SimlFileBuilder.checkpoint_file(
        TEST_DIR / "snapshot_epoch_1000.pth"
    )
    builder.create_loaded(checkpoint_file.file_path)

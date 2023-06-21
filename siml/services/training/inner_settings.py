from typing import Any, Union

import pydantic

from siml.setting import MainSetting, TrainerSetting
from siml.path_like_objects import SimlDirectory


# HACK
# In the future, this setting is merged to setting.TrainerSetting
class InnerTrainingSetting(pydantic.BaseModel):
    # HACK: To Avoid recursive validation by pydantic, Any type is used.
    main_settings: Union[MainSetting, Any]

    class Config:
        arbitrary_types_allowed = True

    @property
    def trainer_setting(self) -> TrainerSetting:
        return self.main_settings.trainer

    def __post_init__(self):
        self._check_restart_and_pretrain()
        self._load_restart_settings()

    def _check_restart_and_pretrain(self):
        if self.trainer_setting.restart_directory is not None \
                and self.trainer_setting.pretrain_directory is not None:
            raise ValueError(
                'Restart directory and pretrain directory cannot be specified '
                'at the same time.'
            )

    def _load_restart_settings(self, only_model: bool = False) -> None:
        if self.trainer_setting.restart_directory is None:
            return

        key = self.main_settings.get_crypt_key()
        restart_directory = self.trainer_setting.restart_directory
        output_directory = self.trainer_setting.output_directory

        siml_dir = SimlDirectory(self.main_settings.trainer.restart_directory)
        restart_setting = MainSetting.read_settings_yaml(
            settings_yaml=siml_dir.find_yaml_file("settings"),
            decrypt_key=key
        )
        if only_model:
            self.main_settings.model = restart_setting.model
        else:
            self.main_settings = restart_setting

        # Overwrite
        # Copy only output_direcoty and crypt key
        self.main_settings.trainer.output_directory = output_directory
        self.main_settings.trainer.model_key = key
        self.main_settings.data.encrypt_key = key
        self.main_settings.inferer.model_key = key

        # restart
        self.main_settings.trainer.restart_directory = restart_directory
        self.main_settings.trainer.pretrain_directory = None

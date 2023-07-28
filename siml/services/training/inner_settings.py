from typing import Any, Union

import pydantic
import pydantic.dataclasses as dc

from siml.path_like_objects import SimlDirectory
from siml.setting import MainSetting, TrainerSetting


# HACK
# In the future, this setting is merged to setting.TrainerSetting
@dc.dataclass(init=True)
class InnerTrainingSetting:
    # HACK: To Avoid recursive validation by pydantic, Any type is used.
    main_setting: Union[MainSetting, Any]

    class Config:
        arbitrary_types_allowed = True

    @pydantic.root_validator
    def _check_not_empty_variables_names(cls, values):
        # HACK: Check in the training settings
        main_setting: MainSetting = values["main_setting"]
        if len(main_setting.trainer.input_names) == 0:
            raise ValueError('No input_names fed')
        if len(main_setting.trainer.output_names) == 0:
            raise ValueError('No output_names fed')
        return values

    @property
    def trainer_setting(self) -> TrainerSetting:
        return self.main_setting.trainer

    @property
    def log_file_path(self):
        return self.trainer_setting.output_directory / 'log.csv'

    @property
    def log_dataset_file_path(self):
        return self.trainer_setting.output_directory / 'log_debug_dataset.txt'

    @property
    def loss_figure_path(self):
        return self.trainer_setting.output_directory \
            / f"plot.{self.trainer_setting.figure_format}"

    def __post_init_post_parse__(self):
        self._check_restart_and_pretrain()
        if self.trainer_setting.restart_directory is not None:
            self._load_restart_settings()

    def _check_restart_and_pretrain(self):
        if self.trainer_setting.restart_directory is not None \
                and self.trainer_setting.pretrain_directory is not None:
            raise ValueError(
                'Restart directory and pretrain directory cannot be specified '
                'at the same time.'
                'pretrain_directory: '
                f'{self.trainer_setting.pretrain_directory},'
                f'restart directory: {self.trainer_setting.restart_directory}'
            )

    def _load_restart_settings(self, only_model: bool = False) -> None:
        key = self.main_setting.get_crypt_key()
        restart_directory = self.trainer_setting.restart_directory
        output_directory = self.trainer_setting.output_directory

        siml_dir = SimlDirectory(self.main_setting.trainer.restart_directory)
        restart_setting = MainSetting.read_settings_yaml(
            settings_yaml=siml_dir.find_yaml_file("settings").file_path,
            decrypt_key=key
        )
        if only_model:
            self.main_setting.model = restart_setting.model
        else:
            self.main_setting = restart_setting

        # Overwrite
        # Copy only output_direcoty and crypt key
        self.main_setting.trainer.output_directory = output_directory
        self.main_setting.trainer.model_key = key
        self.main_setting.data.encrypt_key = key
        self.main_setting.inferer.model_key = key

        # restart
        self.main_setting.trainer.restart_directory = restart_directory
        self.main_setting.trainer.pretrain_directory = None

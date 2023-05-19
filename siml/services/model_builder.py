import pathlib
from collections import OrderedDict
from typing import Optional

from siml import data_parallel, networks, setting
from siml.path_like_objects import ISimlCheckpointFile, SimlFileBuilder
from siml.services.environment import ModelEnvironmentSetting


class ModelBuilder():
    def __init__(
        self,
        model_setting: setting.ModelSetting,
        trainer_setting: setting.TrainerSetting,
        env_setting: ModelEnvironmentSetting
    ) -> None:
        self._model_setting = model_setting
        self._trainer_setting = trainer_setting
        self._env_setting = env_setting

    @property
    def state_dict_strict(self) -> bool:
        return self._trainer_setting.state_dict_strict

    def create_initialized(self) -> networks.Network:
        model = networks.Network(self._model_setting, self._trainer_setting)
        if self._env_setting.data_parallel:
            model = data_parallel.DataParallel(model)

        device = self._env_setting.get_device()
        model.to(device)
        return model

    def create_loaded(
        self,
        checkpoint_file: pathlib.Path,
        decrypt_key: Optional[bytes] = None
    ):
        siml_file = SimlFileBuilder.checkpoint_file(checkpoint_file)

        model = self.create_initialized()
        model_state_dict = self._load_model_state_dict(
            siml_file, model=model, decrypt_key=decrypt_key
        )
        model.load_state_dict(
            model_state_dict,
            strict=self.state_dict_strict
        )
        return model

    def _load_model_state_dict(
        self,
        checkpoint_file: ISimlCheckpointFile,
        *,
        model: Optional[networks.Network] = None,
        decrypt_key: Optional[bytes] = None
    ) -> dict:
        device = self._env_setting.get_device()
        checkpoint = checkpoint_file.load(
            device=device,
            decrypt_key=decrypt_key
        )

        if not self.state_dict_strict:
            model_state_dict = checkpoint['model_state_dict']
            return model_state_dict

        if len(model.state_dict()) != \
                len(checkpoint['model_state_dict']):
            raise ValueError('Model parameter length invalid')

        # Convert new state_dict in case DataParallel wraps model
        model_state_dict = OrderedDict({
            k1: checkpoint['model_state_dict'][k2] for k1, k2
            in zip(
                sorted(model.state_dict().keys()),
                sorted(checkpoint['model_state_dict'].keys())
            )
        })
        return model_state_dict

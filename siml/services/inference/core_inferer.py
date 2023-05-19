from typing import Callable, Optional
import pathlib

from ignite.engine import Engine, State
from torch.utils.data import DataLoader

from siml import networks, setting
from siml.inferer import ModelEnvironmentSetting
from siml.loss_operations import ILossCalculator
from siml.services.model_builder import ModelBuilder

from .engine_bulider import InferenceEngineBuilder
from .metrics_builder import MetricsBuilder
from .postprocessing import PostProcessor


class CoreInferer():
    def __init__(
        self,
        trainer_setting: setting.TrainerSetting,
        model_setting: setting.ModelSetting,
        env_setting: ModelEnvironmentSetting,
        snapshot_file: pathlib.Path,
        prepare_batch_function: Callable,
        loss_function: ILossCalculator,
        post_processor: PostProcessor,
        decrypt_key: Optional[bytes] = None
    ) -> None:

        self._trainer_setting = trainer_setting
        self._model_setting = model_setting
        self._env_setting = env_setting
        self._prepare_batch_function = prepare_batch_function
        self._snapshot_file = snapshot_file
        self._loss_function = loss_function
        self._post_processor = post_processor
        self._decrypt_key = decrypt_key

        self._model = self._load_model()
        self.inferer_engine = self._create_engine()

    def _load_model(self) -> networks.Network:
        model_loader = ModelBuilder(
            model_setting=self._model_setting,
            trainer_setting=self._trainer_setting,
            env_setting=self._env_setting
        )
        model = model_loader.create_loaded(
            self._snapshot_file,
            decrypt_key=self._decrypt_key
        )
        return model

    def _create_engine(self) -> Engine:
        bulider = InferenceEngineBuilder(
            env_setting=self._env_setting,
            prepare_batch_function=self._prepare_batch_function,
            non_blocking=self._trainer_setting.non_blocking,
            post_processor=self._post_processor
        )
        evaluator_engine = bulider.create(self._model)

        metrics_builder = MetricsBuilder(
            trainer_setting=self._trainer_setting,
            loss_function=self._loss_function
        )
        dict_metrics = metrics_builder.create()

        for name, metric in dict_metrics.items():
            metric.attach(evaluator_engine, name)
        return evaluator_engine

    def run(
        self,
        data_loader: DataLoader
    ) -> State:

        inference_state = self.inferer_engine.run(data_loader)

        return inference_state

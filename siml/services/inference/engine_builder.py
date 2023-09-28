from typing import Callable
import time

import torch
from ignite.engine import Engine

from siml import networks
from siml.siml_variables import siml_tensor_variables
from siml.services.environment import ModelEnvironmentSetting

from .record_object import RawPredictionRecord
from .postprocessing import PostProcessor


class InferenceEngineBuilder:
    def __init__(
        self,
        env_setting: ModelEnvironmentSetting,
        prepare_batch_function: Callable,
        post_processor: PostProcessor,
        non_blocking: bool,
        inference_start_datetime: str
    ) -> None:
        self._model_env = env_setting
        self._prepare_batch_func = prepare_batch_function
        self._post_processor = post_processor
        self._non_blocking = non_blocking
        self._start_datetime = inference_start_datetime

    def create(
        self,
        model: networks.Network
    ) -> Engine:
        inference_process = self._create_process_function(
            model=model
        )
        evaluator_engine = Engine(inference_process)
        return evaluator_engine

    def _create_process_function(
        self,
        model: networks.Network
    ) -> Callable:

        def _inference_process(engine, batch):
            model.eval()
            x, y = self._prepare_batch_func(
                batch,
                device=self._model_env.get_device(),
                output_device=self._model_env.get_output_device(),
                support_device=self._model_env.get_device(),
                non_blocking=self._non_blocking
            )
            with torch.no_grad():
                start_time = time.time()
                y_pred = model(x)
                end_time = time.time()
                elapsed_time = end_time - start_time

            assert len(batch['data_directories']) == 1

            data_directory = batch['data_directories'][0]
            result = RawPredictionRecord(
                y_pred=siml_tensor_variables(y_pred),
                y=siml_tensor_variables(y),
                x=siml_tensor_variables(x['x']),
                original_shapes=x['original_shapes'],
                data_directory=data_directory,
                inference_time=elapsed_time
            )
            post_result = self._post_processor.convert(
                result, self._start_datetime
            )
            return y_pred, y, {
                "result": result,
                "post_result": post_result
            }
        return _inference_process

from typing import Callable

import ignite
import numpy as np
import torch
import yaml

from siml import util
from siml.siml_variables import siml_tensor_variables
from siml.networks import Network
from siml.services.environment import ModelEnvironmentSetting
from siml.setting import TrainerSetting
from siml.update_functions import IStepUpdateFunction
from siml.loss_operations import ILossCalculator


class TrainerEngineBuilder:
    def __init__(
        self,
        env_setting: ModelEnvironmentSetting,
        prepare_batch_function: Callable,
        trainer_setting: TrainerSetting
    ) -> None:
        self._env_setting = env_setting
        self._prepare_batch_func = prepare_batch_function
        self._trainer_setting = trainer_setting

    def create_supervised_trainer(
        self,
        model: Network,
        optimizer: torch.optim.Optimizer,
        update_function: IStepUpdateFunction
    ) -> ignite.engine.Engine:

        def update_model(engine, batch):
            model.train()
            if self._trainer_setting.time_series_split is None:
                input_device = self._env_setting.get_device()
                support_device = self._env_setting.get_device()
                output_device = self._env_setting.get_output_device()
            else:
                input_device = 'cpu'
                support_device = self._env_setting.get_device()
                output_device = 'cpu'
            x, y = self._prepare_batch_func(
                batch, device=input_device,
                output_device=output_device,
                support_device=support_device,
                non_blocking=self._trainer_setting.non_blocking
            )
            loss = update_function(x, y, model, optimizer)
            if self._trainer_setting.output_stats:
                self.output_stats(model, engine)
            return loss

        return ignite.engine.Engine(update_model)

    def output_stats(
        self,
        model: Network,
        engine: ignite.engine.Engine
    ):
        dict_stats = {
            name: self._calculate_stats(parameter)
            for name, parameter in model.named_parameters()}
        iteration = engine.state.iteration
        epoch = engine.state.epoch
        dict_stats.update({
            'iteration': engine.state.iteration,
            'epoch': engine.state.epoch,
            'epoch_length': engine.state.epoch_length})

        file_name = self._trainer_setting.output_directory \
            / f"stats_epoch{epoch}_iteration{iteration}.yml"
        with open(file_name, 'w') as f:
            yaml.dump(dict_stats, f)
        return dict_stats

    def _calculate_stats(self, parameter):
        dict_stats = {}
        if parameter.grad is not None:
            dict_stats.update(
                self._calculate_tensor_stats(parameter.grad, 'grad_'))
        dict_stats.update(self._calculate_tensor_stats(parameter, ''))
        return dict_stats

    def _calculate_tensor_stats(self, tensor, prefix):
        numpy_tensor = tensor.detach().cpu().numpy()
        abs_numpy_tensor = np.abs(numpy_tensor)
        return {
            f"{prefix}mean": float(np.mean(numpy_tensor)),
            f"{prefix}std": float(np.std(numpy_tensor)),
            f"{prefix}max": float(np.max(numpy_tensor)),
            f"{prefix}min": float(np.min(numpy_tensor)),
            f"{prefix}absmax": float(np.max(abs_numpy_tensor)),
            f"{prefix}absmin": float(np.min(abs_numpy_tensor)),
        }


class EvaluatorEngineBuilder:
    def __init__(
        self,
        env_setting: ModelEnvironmentSetting,
        prepare_batch_function: Callable,
        trainer_setting: TrainerSetting,
        loss_function: ILossCalculator
    ) -> None:
        self._env_setting = env_setting
        self._prepare_batch_func = prepare_batch_function
        self._trainer_setting = trainer_setting
        self._loss = loss_function

    def create_supervised_evaluator(
        self,
        model: Network,
        input_time_series_keys: list[str],
        output_time_series_keys: list[str],
        split_data_if_needed: Callable,
        update_original_shapes: Callable
    ) -> ignite.engine.Engine:

        # HACK: split_data_if_needed, update_original_shapes functions
        # should be implemented in ISimlVariables

        def _inference(engine, batch):
            model.eval()
            if self._trainer_setting.time_series_split_evaluation is None:
                input_device = self._env_setting.get_device()
                support_device = self._env_setting.get_device()
            else:
                input_device = 'cpu'
                support_device = self._env_setting.get_device()

            with torch.no_grad():
                x, y = self._prepare_batch_func(
                    batch, device=input_device,
                    output_device='cpu',
                    support_device=support_device,
                    non_blocking=self._trainer_setting.non_blocking
                )
                split_xs, split_ys = split_data_if_needed(
                    x, y,
                    self._trainer_setting.time_series_split_evaluation)

                y_pred = []
                device = self._env_setting.get_device()
                for split_x in split_xs:
                    siml_x = siml_tensor_variables(split_x['x'])
                    split_x['x'] = siml_x.send(device).get_values()
                    y_pred.append(model(split_x))

                if self._trainer_setting.time_series_split_evaluation is None:
                    original_shapes = x['original_shapes']
                else:
                    cat_x = util.cat_time_series(
                        [split_x['x'] for split_x in split_xs],
                        time_series_keys=input_time_series_keys)
                    original_shapes = update_original_shapes(
                        cat_x, x['original_shapes']
                    )
                y_pred = util.cat_time_series(
                    y_pred, time_series_keys=output_time_series_keys)

                ans_y = util.cat_time_series(
                    split_ys,
                    time_series_keys=output_time_series_keys
                )
                ans_siml_y = siml_tensor_variables(ans_y)
                output_device = self._env_setting.get_output_device()
                y = ans_siml_y.send(output_device).get_values()
                return y_pred, y, {
                    'original_shapes': original_shapes,
                    'model': model
                }

        evaluator_engine = ignite.engine.Engine(_inference)

        metrics = {'loss': ignite.metrics.Loss(self._loss)}
        for loss_key in model.get_loss_keys():
            metrics.update({loss_key: self._generate_metric(loss_key)})

        for name, metric in metrics.items():
            metric.attach(evaluator_engine, name)
        return evaluator_engine

    def _generate_metric(self, loss_key: str):
        if 'residual' in loss_key:
            def gather_loss_key(x):
                model = x[2]['model']
                return model.get_losses()[loss_key]
            metric = ignite.metrics.Average(output_transform=gather_loss_key)
            return metric

        raise ValueError(f"Unexpected loss type: {loss_key}")

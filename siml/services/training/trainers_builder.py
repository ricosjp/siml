from __future__ import annotations
from typing import Callable, Optional

from ignite.engine import Engine
import torch
from torch.optim import Optimizer

from .tensor_spliter import TensorSpliter
from siml import networks, setting
from siml import update_functions
from siml.services.environment import ModelEnvironmentSetting
from siml.services.model_selector import ModelSelectorBuilder
from siml.loss_operations import ILossCalculator
from siml.services.model_builder import ModelBuilder
from siml.services.training.engine_builder import (
    TrainerEngineBuilder, EvaluatorEngineBuilder
)
from siml.services.training.training_logger import TrainDataDebugLogger


class UpdateFunctionSelector:
    def __init__(
        self,
        trainer_setting: setting.TrainerSetting,
        env_setting: ModelEnvironmentSetting,
        loss_function: ILossCalculator,
        spliter: TensorSpliter
    ) -> None:
        self._trainer_setting = trainer_setting
        self._env_setting = env_setting
        self._loss_function = loss_function
        self._spliter = spliter

    def select(
        self
    ) -> update_functions.IStepUpdateFunction:
        element_wise = self._trainer_setting.determine_element_wise()
        if self._trainer_setting.pseudo_batch_size >= 1:
            return update_functions.PseudoBatchStep(
                batch_size=self._trainer_setting.pseudo_batch_size,
                loss_func=self._loss_function,
                other_loss_func=self._calculate_other_loss,
                split_data_func=self._spliter._split_data_if_needed,
                device=self._env_setting.get_device(),
                output_device=self._env_setting.get_output_device(),
                loss_slice=self._trainer_setting.loss_slice,
                time_series_split=self._trainer_setting.time_series_split,
                clip_grad_norm=self._trainer_setting.clip_grad_norm,
                clip_grad_value=self._trainer_setting.clip_grad_value
            )
        if not element_wise \
                and self._trainer_setting.element_batch_size > 0:
            return update_functions.ElementBatchUpdate(
                element_batch_size=self._trainer_setting.element_batch_size,
                loss_func=self._loss_function,
                other_loss_func=self._calculate_other_loss,
                split_data_func=self._spliter._split_data_if_needed,
                clip_grad_norm=self._trainer_setting.clip_grad_norm,
                clip_grad_value=self._trainer_setting.clip_grad_value
            )

        return update_functions.StandardUpdate(
            loss_func=self._loss_function,
            other_loss_func=self._calculate_other_loss,
            split_data_func=self._spliter._split_data_if_needed,
            device=self._env_setting.get_device(),
            output_device=self._env_setting.get_output_device(),
            loss_slice=self._trainer_setting.loss_slice,
            time_series_split=self._trainer_setting.time_series_split,
            clip_grad_norm=self._trainer_setting.clip_grad_norm,
            clip_grad_value=self._trainer_setting.clip_grad_value
        )

    def _calculate_other_loss(self, model, y_pred, y, original_shapes=None):

        # HACK: This method should be implemented as a ILossCalculator
        # In order to do so, loss_coeffs and losses are enable to be gained
        # from model_settings

        if original_shapes is None:
            original_shapes = [y_pred.shape]
        loss = 0.
        loss_coeffs = model.get_loss_coeffs()
        for loss_key, loss_value in model.get_losses().items():
            if 'residual' in loss_key:
                loss += loss_value * loss_coeffs[loss_key]
            else:
                raise ValueError(f"Unexpected loss_key: {loss_key}")
        return loss


class TrainersBuilder():
    def __init__(
        self,
        trainer_setting: setting.TrainerSetting,
        model_setting: setting.ModelSetting,
        env_setting: ModelEnvironmentSetting,
        prepare_batch_function: Callable,
        loss_function: ILossCalculator,
        decrypt_key: Optional[bytes] = None,
        debug_logger: Optional[TrainDataDebugLogger] = None
    ) -> None:

        self._trainer_setting = trainer_setting
        self._model_setting = model_setting
        self._env_setting = env_setting
        self._prepare_batch_function = prepare_batch_function
        self._loss_function = loss_function
        self._decrypt_key = decrypt_key

        self._spliter = TensorSpliter(
            input_time_series_keys=self._trainer_setting.get_input_time_series_keys(),  # NOQA
            output_time_series_keys=self._trainer_setting.get_output_time_series_keys()  # NOQA
        )
        self._update_func_creator = UpdateFunctionSelector(
            trainer_setting=self._trainer_setting,
            env_setting=self._env_setting,
            loss_function=self._loss_function,
            spliter=self._spliter
        )

        self._trainer_builder = TrainerEngineBuilder(
            env_setting=self._env_setting,
            prepare_batch_function=self._prepare_batch_function,
            trainer_setting=self._trainer_setting,
            debug_logger=debug_logger
        )
        self._evaluator_builder = EvaluatorEngineBuilder(
            env_setting=self._env_setting,
            prepare_batch_function=self._prepare_batch_function,
            trainer_setting=self._trainer_setting,
            loss_function=self._loss_function,
            spliter=self._spliter
        )

    def create(
        self,
        epoch_length: Optional[int] = None
    ) -> tuple[Engine, Engine, networks.Network, Optimizer]:
        model = self.create_model()
        optimizer = self.create_optimizer(model)
        trainer, evaluator = self.create_engines(model, optimizer)
        self.restore_restart_state_if_needed(
            model, optimizer, trainer, epoch_length
        )
        return trainer, evaluator, model, optimizer

    def create_engines(
        self,
        model: networks.Network,
        optimizer: Optimizer
    ) -> tuple[Engine, Engine]:
        update_function = self._update_func_creator.select()
        trainer = self._trainer_builder.create_supervised_trainer(
            model=model,
            optimizer=optimizer,
            update_function=update_function
        )
        evaluator = self._evaluator_builder.create_supervised_evaluator(
            model=model
        )
        return trainer, evaluator

    def create_model(self):
        model_loader = ModelBuilder(
            model_setting=self._model_setting,
            trainer_setting=self._trainer_setting,
            env_setting=self._env_setting
        )
        if self._trainer_setting.pretrain_directory is None:
            return model_loader.create_initialized()

        selector = ModelSelectorBuilder.create(
            self._trainer_setting.snapshot_choise_method
        )
        snapshot_file = selector.select_model(
            self._trainer_setting.pretrain_directory
        )

        return model_loader.create_loaded(
            snapshot_file.file_path,
            decrypt_key=self._decrypt_key
        )

    def create_optimizer(
        self,
        model: networks.Network
    ) -> Optimizer:
        optimizer_name = self._trainer_setting.optimizer.lower()
        if optimizer_name == 'adam':
            return torch.optim.Adam(
                model.parameters(),
                **self._trainer_setting.optimizer_setting)

        if optimizer_name == 'adamw':
            return torch.optim.AdamW(
                model.parameters(),
                **self._trainer_setting.optimizer_setting)

        raise ValueError(f"Unknown optimizer name: {optimizer_name}")

    def restore_restart_state_if_needed(
        self,
        model: networks.Network,
        optimizer: Optimizer,
        trainer: Engine,
        epoch_length: int
    ) -> None:
        if self._trainer_setting.restart_directory is None:
            return

        self._restore_restart_state(
            model, optimizer, trainer, epoch_length
        )
        return

    def _restore_restart_state(
        self,
        model: networks.Network,
        optimizer: Optimizer,
        trainer: Engine,
        epoch_length: int
    ):
        selector = ModelSelectorBuilder.create('latest')
        snapshot_file = selector.select_model(
            self._trainer_setting.restart_directory
        )
        checkpoint = snapshot_file.load(
            device=self._env_setting.get_device(),
            decrypt_key=self._decrypt_key
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.load_state_dict({
            'epoch': checkpoint['epoch'],
            'validation_loss': checkpoint['validation_loss'],
            'seed': self._trainer_setting.seed,
            'max_epochs': self._trainer_setting.n_epoch,
            'epoch_length': epoch_length
        })

        if self._trainer_setting.n_epoch == checkpoint['epoch']:
            raise FileExistsError(
                "Checkpoint at last epoch exists. "
                "Model to restart has already finished"
            )

        # self.loss = checkpoint['loss']
        print(f"{snapshot_file.file_path} loaded for restart.")
        return

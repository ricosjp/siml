from typing import Callable, Union, Optional

import numpy as np
from ignite.engine import Engine, State
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from siml.loss_operations import LossCalculatorBuilder, ILossCalculator
from siml.networks import Network
from siml.services.environment import ModelEnvironmentSetting
from siml.services.training import (InnerTrainingSetting,
                                    SimlTrainingConsoleLogger,
                                    SimlTrainingFileLogger)
from siml.services.training.training_logger import TrainDataDebugLogger
from siml.services.training.data_loader_builder import DataLoaderBuilder
from siml.services.training.events_assigners import (TrainerEventsAssigner,
                                                     ValidationEventsAssigner)
from siml.services.training.trainers_builder import TrainersBuilder
from siml.utils.timer import SimlStopWatch

from . import datasets, setting, util


class Trainer:

    def __init__(self,
                 main_settings: setting.MainSetting,
                 *,
                 optuna_trial=None,
                 user_loss_function_dic:
                 dict[str, Callable[[Tensor, Tensor], Tensor]] = None):
        """Initialize SimlManager object.

        Parameters
        ----------
        settings: siml.setting.MainSetting object or pathlib.Path
            Setting descriptions.
        model: siml.networks.Network object
            Model to be trained.
        optuna_trial: optuna.Trial
            Optuna trial object. Used for pruning.

        Returns
        --------
        None
        """

        if not isinstance(main_settings, setting.MainSetting):
            raise ValueError(
                f"Unknown type for settings: {main_settings.__class__}")

        self.user_loss_function_dic = user_loss_function_dic

        self._inner_setting = InnerTrainingSetting(main_setting=main_settings)
        # HACK: temporarly hack. Better to handle as a inner setting.
        self.setting = self._inner_setting.main_setting

        self.optuna_trial = optuna_trial
        self._env_setting = self._create_model_env_setting()
        self._collate_fn = self._create_collate_fn()
        self._data_loader_builder = DataLoaderBuilder(
            main_setting=self.setting,
            collate_fn=self._collate_fn,
            decrypt_key=self.setting.get_crypt_key()
        )
        self._loss_calculator = LossCalculatorBuilder.create(
            trainer_setting=self.setting.trainer,
            user_loss_function_dic=self.user_loss_function_dic
        )
        self._debug_logger = self._setup_debug_logger()
        self._trainers_builder = self._create_trainers_builder(
            prepare_batch=self._collate_fn.prepare_batch,
            loss_calculator=self._loss_calculator,
            debug_logger=self._debug_logger
        )

        # SET IN _initialize_state()
        self.train_loader: DataLoader = None
        self.validation_loader: DataLoader = None
        self.test_loader: DataLoader = None
        self._model: Network = None
        self._optimizer: Optimizer = None
        self._trainer: Engine = None
        self._evaluator: Engine = None
        self._console_logger: SimlTrainingConsoleLogger = None
        self._file_logger: SimlTrainingFileLogger = None
        self._stop_watch: SimlStopWatch = None

        # Prepare
        self._env_setting.set_seed()
        self._initialize_state()

    def train(self, draw_model: bool = True) -> float:
        """Start training

        Parameters
        ----------
        draw_model : bool, optional
            If True, output figure of models, by default True

        Returns
        -------
        float
            loss for validaiton data
        """
        self._prepare_files_and_dirs(draw_model=draw_model)
        validation_loss = self._run_training()
        return validation_loss

    def evaluate(
        self,
        evaluate_test: bool = False,
        load_best_model: bool = False
    ) -> tuple[State, Union[State, None], Union[State, None]]:
        """Evaluate model performance

        Parameters
        ----------
        evaluate_test : bool, optional
            If True, evaluation by test dataset is performed, by default False
        load_best_model : bool, optional
            If True, best model is used to evaluate, by default False

        Returns
        -------
        tuple[State, Union[State, None], Union[State, None]]
            ignite State objects for train, validation and test dataset
        """

        if load_best_model:
            self.setting.trainer.pretrain_directory \
                = self.setting.trainer.output_directory
            self._initialize_state()
        train_state = self._evaluator.run(self.train_loader)
        if len(self.validation_loader) > 0:
            validation_state = self._evaluator.run(self.validation_loader)
        else:
            validation_state = None

        if evaluate_test:
            test_state = self._evaluator.run(self.test_loader)
        else:
            test_state = None
        return train_state, validation_state, test_state

    def _initialize_state(self):
        # Initialize state
        self.train_loader, self.validation_loader, self.test_loader = \
            self._data_loader_builder.create()
        self._trainer, self._evaluator, self._model, self._optimizer =\
            self._trainers_builder.create(epoch_length=len(self.train_loader))

        self._console_logger, self._file_logger \
            = self._setup_loggers(self._model)
        self._stop_watch = self._setup_stopwatch(self._file_logger)

        # HACK: NEED to be called last
        self._assign_trainer_events(self._trainer)
        self._assign_evaluator_events(self._evaluator)

    def _prepare_files_and_dirs(self, draw_model: bool = True):
        # setup Directory
        self.setting.trainer.output_directory.mkdir(
            parents=True, exist_ok=True
        )
        if self.setting.trainer.draw_network and draw_model:
            self._model.draw(self.setting.trainer.output_directory)
        self._write_settings_yaml()

    def _write_settings_yaml(self):
        overwrite_restart_mode = self.setting.trainer.overwrite_restart_mode
        yaml_file_name = f"restart_settings_{util.date_string()}.yml" \
            if overwrite_restart_mode else "settings.yml"
        setting.write_yaml(
            self.setting,
            self.setting.trainer.output_directory / yaml_file_name,
            key=self.setting.trainer.model_key
        )

    def _run_training(self):
        # start logging
        self._console_logger.output_header()
        self._file_logger.write_header_if_needed()
        self._stop_watch.start()

        self._trainer.run(
            self.train_loader,
            max_epochs=self.setting.trainer.n_epoch
        )
        df = self._file_logger.read_history()
        validation_loss = np.min(df['validation_loss'])
        return validation_loss

    def _setup_loggers(
        self,
        model: Network
    ) -> tuple[SimlTrainingConsoleLogger, SimlTrainingFileLogger]:
        console_logger = SimlTrainingConsoleLogger(
            display_margin=self.setting.trainer.display_mergin,
            loss_keys=model.get_loss_keys()
        )
        file_logger = SimlTrainingFileLogger(
            file_path=self._inner_setting.log_file_path,
            loss_figure_path=self._inner_setting.loss_figure_path,
            loss_keys=model.get_loss_keys(),
            continue_mode=self.setting.trainer.overwrite_restart_mode
        )

        return console_logger, file_logger

    def _setup_debug_logger(self) -> Union[TrainDataDebugLogger, None]:
        if self._inner_setting.trainer_setting.debug_dataset:
            file_path = self._inner_setting.log_dataset_file_path
            debug_logger = TrainDataDebugLogger(file_path)
        else:
            debug_logger = None
        return debug_logger

    def _setup_stopwatch(self, file_logger: SimlTrainingFileLogger):
        time_offset = file_logger.read_offset_start_time()
        stop_watch = SimlStopWatch(offset=time_offset)
        return stop_watch

    def _assign_trainer_events(self, trainer: Engine) -> None:
        train_events = TrainerEventsAssigner(
            trainer_setting=self.setting.trainer,
            file_logger=self._file_logger,
            console_logger=self._console_logger,
            train_loader=self.train_loader,
            validation_loader=self.validation_loader,
            evaluator=self._evaluator,
            model=self._model,
            optimizer=self._optimizer,
            timer=self._stop_watch,
            debug_logger=self._debug_logger
        )
        train_events.assign_handlers(trainer)

    def _assign_evaluator_events(self, evaluator: Engine) -> None:
        evaluator_events = ValidationEventsAssigner(
            file_logger=self._file_logger,
            console_logger=self._console_logger,
            log_trigger_epoch=self.setting.trainer.log_trigger_epoch,
            train_loader=self.train_loader,
            validation_loader=self.validation_loader,
            trainer=self._trainer,
            model=self._model,
            trainer_setting=self.setting.trainer,
            timer=self._stop_watch
        )
        evaluator_events.assign_handlers(
            evaluator, self.optuna_trial
        )

    def _create_model_env_setting(self) -> ModelEnvironmentSetting:
        trainer_setting = self.setting.trainer
        _model_env = ModelEnvironmentSetting(
            gpu_id=trainer_setting.gpu_id,
            seed=trainer_setting.seed,
            data_parallel=trainer_setting.data_parallel,
            model_parallel=trainer_setting.model_parallel,
            time_series=trainer_setting.time_series
        )
        return _model_env

    def _create_collate_fn(self):
        tr_setting = self.setting.trainer

        collate_fn = datasets.CollateFunctionGenerator(
            time_series=tr_setting.time_series,
            dict_input=tr_setting.input_is_dict,
            dict_output=tr_setting.output_is_dict,
            use_support=tr_setting.support_inputs,
            element_wise=tr_setting.determine_element_wise(),
            data_parallel=tr_setting.data_parallel,
            input_time_series_keys=tr_setting.get_input_time_series_keys(),
            output_time_series_keys=tr_setting.get_output_time_series_keys(),
            input_time_slices=tr_setting.inputs.time_slice,
            output_time_slices=tr_setting.outputs.time_slice
        )
        return collate_fn

    def _create_trainers_builder(
        self,
        prepare_batch: Callable,
        loss_calculator: ILossCalculator,
        debug_logger: Optional[TrainDataDebugLogger] = None
    ):
        trainers_builder = TrainersBuilder(
            trainer_setting=self.setting.trainer,
            model_setting=self.setting.model,
            env_setting=self._env_setting,
            prepare_batch_function=prepare_batch,
            loss_function=loss_calculator,
            decrypt_key=self.setting.get_crypt_key(),
            debug_logger=debug_logger
        )
        return trainers_builder

import enum
from typing import Optional

import numpy as np
import optuna
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm

from siml.networks import Network
from siml.services.training import (LogRecordItems, SimlTrainingConsoleLogger,
                                    SimlTrainingFileLogger)
from siml.services.training.training_logger import TrainDataDebugLogger
from siml.setting import TrainerSetting
from siml.utils.timer import SimlStopWatch


# Add early stopping
class StopTriggerEvents(enum.Enum):
    EVALUATED = 'evaluated'


class TrainerEventsAssigner:
    def __init__(
        self,
        trainer_setting: TrainerSetting,
        file_logger: SimlTrainingFileLogger,
        console_logger: SimlTrainingConsoleLogger,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        evaluator: Engine,
        model: Network,
        optimizer: Optimizer,
        timer: SimlStopWatch,
        debug_logger: Optional[TrainDataDebugLogger] = None
    ) -> None:
        self._file_logger = file_logger
        self._console_logger = console_logger
        self._evaluator = evaluator
        self._train_loader = train_loader
        self._validation_loader = validation_loader
        self._model = model
        self._trainer_setting = trainer_setting
        self._timer = timer
        self._optimizer = optimizer
        self._debug_logger = debug_logger

        self._desc: str = "loss: {:.5e}"
        self._trick = 1
        self._evaluator_tick = 1

        total = len(train_loader) + self._trainer_setting.log_trigger_epoch
        self._pbar = _create_pbar(total, self._desc.format(0))

    def assign_handlers(self, trainer: Engine) -> None:
        """Assign handlers to trainer engine

        Parameters
        ----------
        trainer : Engine
            trainer engine object
        """
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED,
            self.log_training_loss
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(
                every=self._trainer_setting.log_trigger_epoch
            ),
            self.log_training_results
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(
                every=self._trainer_setting.stop_trigger_epoch
            ),
            self.fire_stop_trigger
        )

    def log_training_loss(self, engine: Engine):
        self._pbar.desc = self._desc.format(engine.state.output)
        self._pbar.update(self._trick)
        return

    def log_training_results(self, engine: Engine):
        self._pbar.close()

        train_loss, train_other_losses = self._evaluate_losses(
            self._evaluator,
            self._train_loader
        )
        validation_loss, validation_other_losses = \
            self._evaluate_losses(
                self._evaluator,
                self._validation_loader
            )
        log_record = LogRecordItems(
            epoch=engine.state.epoch,
            train_loss=train_loss,
            train_other_losses=train_other_losses,
            validation_loss=validation_loss,
            validation_other_losses=validation_other_losses,
            elapsed_time=self._timer.watch()
        )

        # Print log
        tqdm.write(self._console_logger.output(log_record))

        self._pbar.n = self._pbar.last_print_n = 0

        # Save checkpoint
        self._file_logger.save_model(
            epoch=engine.state.epoch,
            model=self._model,
            optimizer=self._optimizer,
            validation_loss=validation_loss,
            trainer_setting=self._trainer_setting
        )

        if self._debug_logger is not None:
            self._debug_logger.write_epoch(engine.state.epoch)
        # Write log
        self._file_logger.write(log_record)

        # Plot
        self._file_logger.save_figure()
        return

    def fire_stop_trigger(self, engine):
        self._evaluator.fire_event(StopTriggerEvents.EVALUATED)
        return

    def _evaluate_losses(
        self,
        evaluator: Engine,
        data_loader: DataLoader
    ) -> tuple[float, dict[str, float]]:

        if len(data_loader) <= 0:
            return np.nan, {}
        evaluator.run(data_loader)
        loss = evaluator.state.metrics['loss']
        other_losses = {
            k: v for k, v in evaluator.state.metrics.items()
            if k != 'loss'
        }
        return loss, other_losses


class ValidationEventsAssigner:
    def __init__(
        self,
        file_logger: SimlTrainingFileLogger,
        console_logger: SimlTrainingConsoleLogger,
        log_trigger_epoch: int,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        trainer: Engine,
        model: Network,
        trainer_setting: TrainerSetting,
        timer: SimlStopWatch
    ) -> None:
        self._file_logger = file_logger
        self._console_logger = console_logger
        self._log_trigger_epoch = log_trigger_epoch
        self._trainer = trainer
        self._train_loader = train_loader
        self._validation_loader = validation_loader
        self._model = model
        self._trainer_setting = trainer_setting
        self._timer = timer

        self._desc: str = "evaluating"
        self._trick = 1
        self._evaluator_tick = 1

        total = len(train_loader) + len(validation_loader)
        self._pbar = _create_pbar(total, self._desc)

    def assign_handlers(
        self,
        evaluator: Engine,
        optuna_trial: bool
    ) -> None:
        """Assign handlers to evaluator engine

        Parameters
        ----------
        evaluator : Engine
            evaluator engine
        optuna_trial : bool
            If true, run optuna
        """
        evaluator.add_event_handler(
            Events.ITERATION_COMPLETED(every=self._evaluator_tick),
            self.log_evaluation
        )
        self._register_early_stopping(evaluator)
        if optuna_trial:
            self._register_pruning_handlers(
                optuna_trial=optuna_trial, evaluator=evaluator
            )

    def log_evaluation(self, engine):
        self._pbar.desc = self._desc
        self._pbar.update(self._evaluator_tick)
        return

    def _register_early_stopping(self, evaluator: Engine):
        evaluator.register_events(*StopTriggerEvents)
        early_stopping_handler = EarlyStopping(
            patience=self._trainer_setting.patience,
            score_function=self.score_function,
            trainer=self._trainer
        )
        evaluator.add_event_handler(
            StopTriggerEvents.EVALUATED,
            early_stopping_handler
        )

    def score_function(self, engine: Engine) -> float:
        return -1.0 * engine.state.metrics['loss']

    def _register_pruning_handlers(
        self, optuna_trial: bool, evaluator: Engine
    ):
        # Add pruning setting
        pruning_handler = optuna.integration.PyTorchIgnitePruningHandler(
            optuna_trial, 'loss', self._trainer)
        evaluator.add_event_handler(
            StopTriggerEvents.EVALUATED, pruning_handler)


def _create_pbar(total: int, desc: str) -> tqdm:
    return tqdm(
        initial=0,
        leave=False,
        total=total,
        desc=desc,
        ncols=80,
        ascii=True
    )

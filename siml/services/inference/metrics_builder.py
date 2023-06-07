from __future__ import annotations
from typing import Union

import numpy as np
import torch
from ignite.metrics.metric import Metric, reinit__is_reduced

from siml import setting
from siml.loss_operations import ILossCalculator

from .record_object import RawPredictionRecord, PredictionRecord


class MetricsBuilder():
    def __init__(
        self,
        trainer_setting: setting.TrainerSetting,
        loss_function: ILossCalculator
    ) -> None:

        self._trainer_setting = trainer_setting
        self.loss_function = loss_function

    def create(self) -> dict[str, Metric]:
        metrics = {
            'post_results': PostResultsMetrics(),
            'loss': LossMetrics(
                loss_function=self.loss_function
            ),
            'raw_loss': RawLossMetrics(
                trainer_setting=self._trainer_setting,
                loss_function=self.loss_function
            )
        }
        return metrics


class PostResultsMetrics(Metric):
    def __init__(
        self
    ) -> None:
        super().__init__()

    @reinit__is_reduced
    def reset(self):
        self._results = []
        return

    @reinit__is_reduced
    def update(self, output):
        record: PredictionRecord = output[2]["post_result"]
        self._results.append(record)
        return

    def compute(self):
        return self._results


class LossMetrics(Metric):
    def __init__(
        self,
        loss_function: ILossCalculator
    ) -> None:
        super().__init__()
        self.loss_function = loss_function

    @reinit__is_reduced
    def reset(self):
        self._results = []
        return

    @reinit__is_reduced
    def update(self, output):
        y_pred = output[0]
        y = output[1]
        record: RawPredictionRecord = output[2]["result"]
        loss = self.loss_function(
            y_pred,
            y,
            original_shapes=record.original_shapes
        )
        if loss is not None:
            print(f"              Loss: {loss}")

            loss = loss.cpu().detach().numpy().item()
        self._results.append(loss)
        return

    def compute(self):
        return self._results


class RawLossMetrics(Metric):

    def __init__(
        self,
        trainer_setting: setting.TrainerSetting,
        loss_function: ILossCalculator
    ) -> None:
        super().__init__()
        self._trainer_setting = trainer_setting
        self.loss_function = loss_function

    @reinit__is_reduced
    def reset(self):
        self._results = []
        return

    @reinit__is_reduced
    def update(self, output):
        post_result: PredictionRecord = output[2]["post_result"]
        raw_loss = self._compute_raw_loss(
            post_result.dict_answer,
            post_result.dict_y,
            post_result.original_shapes
        )

        if raw_loss is not None:
            raw_loss = raw_loss.item()
        self._results.append(raw_loss)
        return

    def compute(self):
        return self._results

    def _compute_raw_loss(
        self,
        dict_answer: dict[str, np.ndarray],
        dict_y: dict[str, np.ndarray],
        original_shapes: tuple = None
    ) -> Union[np.ndarray, None]:

        if dict_answer is None:
            return None  # No answer

        y_keys = dict_y.keys()
        if not np.all([y_key in dict_answer for y_key in y_keys]):
            return None  # No answer

        if isinstance(self._trainer_setting.output_names, dict):
            output_names = self._trainer_setting.output_names
            y_raw_pred = self._reshape_dict(output_names, dict_y)
            y_raw_answer = self._reshape_dict(output_names, dict_answer)
        else:
            y_raw_pred = torch.from_numpy(
                np.concatenate(
                    [dict_y[k] for k in dict_y.keys()],
                    axis=-1
                )
            )
            y_raw_answer = torch.from_numpy(
                np.concatenate(
                    [dict_answer[k] for k in dict_y.keys()],
                    axis=-1
                )
            )

        raw_loss = self.loss_function(
            y_raw_pred, y_raw_answer, original_shapes=original_shapes)
        if raw_loss is None:
            return None
        else:
            return raw_loss.numpy()

    def _reshape_dict(
        self,
        dict_names: list[str],
        data_dict: dict
    ) -> dict[str: torch.Tensor]:
        return {
            key:
            torch.from_numpy(
                np.concatenate([
                    data_dict[variable_name] for variable_name in value
                ])
            )
            for key, value in dict_names.items()
            if key in data_dict.keys()}

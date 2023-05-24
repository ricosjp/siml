from typing import Callable, Optional

import torch

from siml import setting
from siml.siml_variables import siml_tensor_variables

from .loss_calculator import ILossCalculator, LossCalculator


class LossCalculatorBuilder:
    @staticmethod
    def create(
        trainer_setting: setting.TrainerSetting,
        *,
        pad: bool = False,
        allow_no_answer: bool = False,
        user_loss_function_dic: Optional[dict[str, Callable]] = None
    ) -> ILossCalculator:
        if pad:
            raise ValueError('pad = True is no longer supported')

        loss_setting = trainer_setting.loss_function
        output_is_dict = isinstance(
            trainer_setting.outputs.variables, dict)
        loss_calculator = LossCalculator(
            loss_setting=loss_setting,
            output_is_dict=output_is_dict,
            time_series=trainer_setting.time_series,
            output_skips=trainer_setting.output_skips,
            output_dims=trainer_setting.output_dims,
            user_loss_function_dic=user_loss_function_dic
        )
        if not allow_no_answer:
            return loss_calculator

        loss_func = LossCalculatorNoAnswer(loss_calculator)
        return loss_func


class LossCalculatorNoAnswer(ILossCalculator):
    def __init__(self, loss_calculator: ILossCalculator):
        self._loss_calculator = loss_calculator

    def __call__(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        original_shapes: Optional[tuple] = None,
        **kwargs
    ) -> torch.Tensor:
        try:
            if y is None or len(y) == 0:
                return None

            siml_tensor = siml_tensor_variables(y)
            if siml_tensor.min_len() == 0:
                return None

            return self._loss_calculator(
                y_pred, y, original_shapes=original_shapes
            )

        except Exception:
            print('Skip loss computation.')
            return None

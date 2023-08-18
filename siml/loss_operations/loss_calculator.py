import abc
from typing import Callable, Optional, Union

import numpy as np
import torch
from torch import Tensor

from siml import util
from siml.base.siml_enums import LossType

from .loss_assignment import ILossAssignment, LossAssignmentCreator
from .loss_selector import LossFunctionSelector


class ILossCalculator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        original_shapes: Optional[tuple] = None,
        **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def calculate_loss_details(
        self, y_pred, y, original_shapes=None
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError()


class LossCalculator(ILossCalculator):

    def __init__(
        self,
        *,
        loss_setting: Union[dict, str] = LossType.MSE.value,
        time_series: bool = False,
        output_is_dict: bool = False,
        output_skips=None,
        output_dims=None,
        user_loss_function_dic:
        dict[str, Callable[[Tensor, Tensor], Tensor]] = None,
        loss_weights: Optional[dict[str, float]] = None
    ) -> None:

        self.loss_assignment = LossAssignmentCreator.create(loss_setting)
        self.loss_core = CoreLossCalculator(
            loss_assignment=self.loss_assignment,
            user_loss_function_dic=user_loss_function_dic,
            loss_weights=loss_weights
        )

        self.output_is_dict = output_is_dict
        self.output_dims = output_dims
        self.time_series = time_series

        self.mask_function = util.VariableMask(
            output_skips, output_dims, output_is_dict)

    def __call__(
            self, y_pred, y, original_shapes=None, **kwargs) -> Tensor:

        if self.time_series:
            return self._loss_function_time_with_padding(
                y_pred, y, original_shapes
            )

        if self.output_is_dict:
            return self._loss_function_dict(y_pred, y, original_shapes)

        return self._loss_function_without_padding(y_pred, y, original_shapes)

    def calculate_loss_details(
        self, y_pred, y, original_shapes=None
    ) -> dict[str, np.ndarray]:

        if not self.output_is_dict:
            return {}

        masked_y_pred, masked_y, masked_keys = self.mask_function(
            y_pred,
            y,
            with_key_names=True)
        name2loss = {
            key: self.loss_core(
                myp.view(my.shape), my, key
            ).detach().cpu().numpy()
            for myp, my, key in zip(masked_y_pred, masked_y, masked_keys)
        }
        return name2loss

    def _loss_function_dict(self, y_pred, y, original_shapes=None) -> Tensor:
        masked_y_pred, masked_y, masked_keys = self.mask_function(
            y_pred,
            y,
            with_key_names=True)
        return torch.mean(torch.stack([
            self.loss_core(myp.view(my.shape), my, key)
            for myp, my, key in zip(masked_y_pred, masked_y, masked_keys)
        ]))

    def _loss_function_without_padding(
            self, y_pred, y, original_shapes=None) -> Tensor:
        return self.loss_core(*self.mask_function(y_pred.view(y.shape), y))

    def _loss_function_time_with_padding(
            self, y_pred, y, original_shapes) -> Tensor:

        split_y_pred = torch.split(
            y_pred, list(original_shapes[:, 1]), dim=1)
        concatenated_y_pred = torch.cat([
            sy[:s].reshape(-1)
            for s, sy in zip(original_shapes[:, 0], split_y_pred)])
        split_y = torch.split(
            y, list(original_shapes[:, 1]), dim=1)
        concatenated_y = torch.cat([
            sy[:s].reshape(-1)
            for s, sy in zip(original_shapes[:, 0], split_y)])
        return self.loss_core(
            *self.mask_function(concatenated_y_pred, concatenated_y))


class CoreLossCalculator():
    """Calculate loss according to variable name and function name
    """

    def __init__(
        self,
        *,
        loss_assignment: ILossAssignment,
        user_loss_function_dic:
        dict[str, Callable[[Tensor, Tensor], Tensor]] = None,
        loss_weights: Optional[dict[str, float]] = None
    ):

        self.loss_selector = LossFunctionSelector(
            loss_assignment,
            user_loss_function_dic=user_loss_function_dic
        )
        self._loss_weights = loss_weights

    def __call__(
        self,
        input_tensor: Tensor,
        target_tensor: Tensor,
        variable_name: str = None
    ) -> Tensor:
        """Calculate loss value

        Args:
            input_tensor (Tensor): tensor of prediction
            target_tensor (Tensor): tensor of target
            variable_name (str, optional):
             name of variable. Defaults to None.

        Returns:
            Tensor: Loss value
        """
        loss_func = self.loss_selector.get_loss_function(variable_name)
        loss_weight = self._get_loss_weight(variable_name)
        return loss_weight * loss_func(input_tensor, target_tensor)

    def _get_loss_weight(self, variable_name: str) -> float:
        if self._loss_weights is None:
            return 1.0

        weight = self._loss_weights.get(variable_name)
        if weight is None:
            raise KeyError(
                "Weights for all variables must be set "
                "when loss weight is defined in settings."
            )

        return weight

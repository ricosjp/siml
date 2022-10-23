from typing import Callable, Union
import torch
from torch import Tensor

from siml import util
from .loss_assignment import ILossAssignment
from .loss_assignment import LossAssignmentCreator
from .loss_selector import LossFunctionSelector


class LossCalculator:

    def __init__(
            self,
            *,
            loss_setting: Union[dict, str] = 'mse',
            time_series: bool = False,
            output_is_dict: bool = False,
            output_skips=None,
            output_dims=None,
            user_loss_function_dic:
            dict[str, Callable[[Tensor, Tensor], Tensor]] = None):

        self.loss_assignment = LossAssignmentCreator.create(loss_setting)
        self.loss_core = CoreLossCalculator(
            loss_assignment=self.loss_assignment,
            user_loss_function_dic=user_loss_function_dic
        )

        self.output_is_dict = output_is_dict
        self.output_dims = output_dims
        self.time_series = time_series

        self.mask_function = util.VariableMask(
            output_skips, output_dims, output_is_dict)

        if self.time_series:
            self.loss = self.loss_function_time_with_padding
        else:
            if self.output_is_dict:
                self.loss = self.loss_function_dict
            else:
                self.loss = self.loss_function_without_padding

        return

    def __call__(self, y_pred, y, original_shapes=None, **kwargs):
        return self.loss(y_pred, y, original_shapes)

    def loss_function_dict(self, y_pred, y, original_shapes=None):
        masked_y_pred, masked_y, masked_keys = self.mask_function(
            y_pred,
            y,
            with_key_names=True)
        return torch.mean(torch.stack([
            self.loss_core(myp.view(my.shape), my, key)
            for myp, my, key in zip(masked_y_pred, masked_y, masked_keys)
        ]))

    def loss_function_without_padding(self, y_pred, y, original_shapes=None):
        return self.loss_core(*self.mask_function(y_pred.view(y.shape), y))

    def loss_function_time_with_padding(self, y_pred, y, original_shapes):
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
            dict[str, Callable[[Tensor, Tensor], Tensor]] = None):

        self.loss_selector = LossFunctionSelector(
            loss_assignment,
            user_loss_function_dic=user_loss_function_dic
        )
        return

    def __call__(self,
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
        return loss_func(input_tensor, target_tensor)

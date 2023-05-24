from torch import Tensor
from typing import Callable
import torch.nn.functional as functional

from siml.base.siml_enums import LossType

from .loss_assignment import ILossAssignment


class LossFunctionSelector():
    """
    Resposibility: Select loss function for each variable name
    """

    def __init__(
        self,
        loss_assignment: ILossAssignment,
        *,
        user_loss_function_dic:
            dict[str, Callable[[Tensor, Tensor], Tensor]] = None):

        self.loss_assignment = loss_assignment
        self.func_name_to_func_obj = self._create_loss_function_dict(
            user_loss_function_dic
        )
        self._check_loss_functions()

    def get_loss_function(
        self,
        variable_name: str
    ) -> Callable[[Tensor, Tensor], Tensor]:
        loss_name = self.loss_assignment[variable_name]
        return self.func_name_to_func_obj[loss_name]

    def _check_loss_functions(self) -> None:
        for loss_name in self.loss_assignment.loss_names:
            if loss_name not in self.func_name_to_func_obj.keys():
                raise ValueError(f"Unknown loss function name: {loss_name}")

    def _create_loss_function_dict(
            self,
            user_loss_function_dic:
            dict[str, Callable[[Tensor, Tensor], Tensor]] = None
    ) -> dict[str, Callable[[Tensor, Tensor], Tensor]]:
        """Create dictionary of which key is function name and\
            value is funciton object.

        Args:
            user_loss_function_dic
             (dict[str, Callable[[Tensor, Tensor], Tensor]], optional):
              Loss function dictionary defined by user. Defaults to None.

        Returns:
            dict[str, Callable[[Tensor, Tensor], Tensor]]:
             Key is function name and value is function object
        """

        name_to_function = {
            LossType.MSE.value: functional.mse_loss
        }

        if user_loss_function_dic is not None:
            name_to_function.update(user_loss_function_dic)

        return name_to_function

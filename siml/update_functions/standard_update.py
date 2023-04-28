from __future__ import annotations
from typing import Any, Callable
import numpy as np
import torch

from siml.networks.network import Network
from siml.siml_variables import siml_tensor_variables
from .update_interface import IStepUpdateFunction


class StandardUpdate(IStepUpdateFunction):
    def __init__(
        self,
        loss_func: Callable,
        other_loss_func: Callable,
        split_data_func: Callable,
        device: str,
        output_device: str,
        loss_slice: slice,
        time_series_split: bool,
        clip_grad_value: float = None,
        clip_grad_norm: float = None
    ) -> None:
        self._loss_func = loss_func
        self._other_loss_func = other_loss_func
        self._split_data_func = split_data_func

        self.device = device
        self.output_device = output_device
        self.loss_slice = loss_slice
        self.time_series_split = time_series_split

        self._clip_grad_value = clip_grad_value
        self._clip_grad_norm = clip_grad_norm

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        model: Network,
        optimizer: torch.optim.Optimizer,
        *args: Any,
        **kwds: Any
    ) -> float:
        split_xs, split_ys = self._split_data_func(
            x, y, self.time_series_split
        )

        loss_value = np.nan
        for split_x, split_y in zip(split_xs, split_ys):
            optimizer.zero_grad()
            siml_x = siml_tensor_variables(split_x['x']).send(self.device)
            siml_y = siml_tensor_variables(split_y).send(self.output_device)

            split_x['x'] = \
                siml_x.get_values()
            split_y = \
                siml_y.get_values()

            split_y_pred = model(split_x)
            siml_y_pred = siml_tensor_variables(split_y_pred)

            loss = self._loss_func(
                siml_y_pred.slice(self.loss_slice).get_values(),
                siml_y.slice(self.loss_slice).get_values(),
                split_x['original_shapes']
            )
            other_loss = self._other_loss_func(
                model,
                siml_y_pred.slice(self.loss_slice).get_values(),
                siml_y.slice(self.loss_slice).get_values(),
                split_x['original_shapes']
            )
            (loss + other_loss).backward()

            loss_value = float(loss)
            del loss
            del other_loss
            model.clip_if_needed()
            model.clip_uniform_if_needed(
                clip_grad_value=self._clip_grad_value,
                clip_grad_norm=self._clip_grad_norm
            )
            optimizer.step()
            model.reset()
        return loss_value

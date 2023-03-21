from __future__ import annotations
from typing import Any, Callable

import torch

from siml.networks.network import Network
from .update_interface import IStepUpdateFunction


class ElementBatchUpdate(IStepUpdateFunction):
    def __init__(
        self,
        element_batch_size: int,
        loss_func: Callable,
        other_loss_func: Callable,
        split_data_func: Callable,
        clip_grad_value: float = None,
        clip_grad_norm: float = None
    ) -> None:
        self._element_batch_size = element_batch_size
        self._loss_func = loss_func
        self._other_loss_func = other_loss_func
        self._split_data_func = split_data_func

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

        y_pred = model(x)

        optimizer.zero_grad()
        split_y_pred = torch.split(
            y_pred, self._element_batch_size)
        split_y = torch.split(
            y, self._element_batch_size)
        for syp, sy in zip(split_y_pred, split_y):
            optimizer.zero_grad()
            loss = self._loss_func(y_pred, y)
            other_loss = self._other_loss_func(model, y_pred, y)
            (loss + other_loss).backward(retain_graph=True)
            loss.backward(retain_graph=True)

        model.clip_if_needed()
        model.clip_uniform_if_needed(
            clip_grad_value=self._clip_grad_value,
            clip_grad_norm=self._clip_grad_norm
        )
        optimizer.step()
        model.reset()

        loss = self._loss_func(y_pred, y)
        return float(loss)

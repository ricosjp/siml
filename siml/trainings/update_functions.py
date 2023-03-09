from __future__ import annotations
from typing import Any, Callable

import abc
import numpy as np
import torch

from siml.networks.network import Network
from siml.trainings.siml_variables import siml_variables


class IStepUpdateFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            model: Network,
            optimizer: torch.optim.Optimizer,
            *args: Any,
            **kwds: Any) -> float:
        raise NotImplementedError()


class Counter():
    def __init__(self, base_value: int):
        assert base_value > 0
        self._value = 0
        self._base = base_value

    def increment(self) -> None:
        self._value += 1
        self._value %= self._base

    @property
    def value(self) -> int:
        return self._value

    @property
    def is_full(self) -> bool:
        return (self._value + 1) == self._base


class PseudoBatchStep(IStepUpdateFunction):
    def __init__(
        self,
        batch_size: int,
        loss_func: Callable,
        other_loss_func: Callable,
        split_data_func: Callable,
        device: str,
        output_device: str,
        loss_slice: slice,
        time_series_split: bool
    ) -> None:
        self.batch_size = batch_size
        self._loss_func = loss_func
        self._other_loss_func = other_loss_func
        self._split_data_func = split_data_func

        self.device = device
        self.output_device = output_device
        self.loss_slice = loss_slice
        self.time_series_split = time_series_split

        # HACK: Incompatible with parallel execution
        self._counter = Counter(batch_size)

    def _allow_zero_grad(self) -> bool:
        return self._counter.value == 0

    def _allow_update(self) -> bool:
        return self._counter.is_full

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
            if self._allow_zero_grad():
                optimizer.zero_grad()

            siml_x = siml_variables(split_x['x'])
            siml_y = siml_variables(split_y)

            split_x['x'] = \
                siml_x.send(self.device).get_value()
            split_y = \
                siml_y.send(self.output_device).get_value()

            split_y_pred = model(split_x)

            siml_y_pred = siml_variables(split_y_pred)

            _loss = self._loss_func(
                siml_y_pred.slice(self.loss_slice).get_value(),
                siml_y.slice(self.loss_slice).get_value(),
                split_x['original_shapes']
            )
            _other_loss = self._other_loss_func(
                model,
                siml_y_pred.slice(self.loss_slice).get_value(),
                siml_y.slice(self.loss_slice).get_value(),
                split_x['original_shapes']
            )

            (_loss + _other_loss).backward()
            # average
            loss_value = float(_loss) / (self._counter.value + 1)
            del _loss
            del _other_loss

            if self._allow_update():
                self._clip_if_needed(model)
                optimizer.step()
                model.reset()

            self._counter.increment()

        return loss_value

    def _clip_if_needed(
            self,
            model: Network,
            clip_grad_value: float = None,
            clip_grad_norm: float = None
    ) -> None:
        model.clip_if_needed()
        if clip_grad_value is not None:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), clip_grad_value
            )
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad_norm
            )

from __future__ import annotations
from typing import Union
import torch


class SimlVariable():
    def __init__(self, value: Union[torch.Tensor, dict, list, SimlVariable]):
        if isinstance(value, SimlVariable):
            self._x = value._x
        else:
            self._x = value
            self._post_init()

    def _post_init(self):
        if isinstance(self._x, torch.Tensor):
            return
        if isinstance(self._x, dict):
            self._x = {k: SimlVariable(v) for k, v in self._x.items()}
            return
        if isinstance(self._x, list):
            self._x = [SimlVariable(v) for v in self._x]
            return

        raise ValueError(f"Invalid format: {self._x.__class__}")

    def get_value(self) -> Union[torch.Tensor, dict, list]:
        if isinstance(self._x, torch.Tensor):
            tmp = self._x
        elif isinstance(self._x, dict):
            tmp = {k: v.get_value() for k, v in self._x.items()}
        elif isinstance(self._x, list):
            tmp = [v.get_value() for v in self._x]
        else:
            raise ValueError(f"Invalid format: {self._x.__class__}")

        return tmp

    def slice(self, loss_slice: slice) -> SimlVariable:
        if isinstance(self._x, torch.Tensor):
            tmp = self._x[loss_slice]
        elif isinstance(self._x, dict):
            tmp = {k: v.slice(loss_slice) for k, v in self._x.items()}
        elif isinstance(self._x, list):
            tmp = [v.slice(loss_slice) for v in self._x]
        else:
            raise ValueError(f"Invalid format: {self._x.__class__}")

        return SimlVariable(tmp)

    def send(self, device: str) -> SimlVariable:
        if isinstance(self._x, torch.Tensor):
            tmp = self._x.to(device)
        elif isinstance(self._x, dict):
            tmp = {k: v.send(device) for k, v in self._x.items()}
        elif isinstance(self._x, list):
            tmp = [v.send(device) for v in self._x]
        else:
            raise ValueError(f"Invalid format: {self._x.__class__}")

        return SimlVariable(tmp)

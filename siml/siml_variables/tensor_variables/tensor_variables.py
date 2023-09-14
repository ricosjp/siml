from __future__ import annotations

from typing import TypeVar, Any

import torch
import numpy as np

BuiltInVars = TypeVar(
    'BuiltInVars',
    torch.Tensor, dict, list
)


class ISimlVariables():
    def __init__(self, value: BuiltInVars) -> None:
        raise NotImplementedError()

    def get_values(self) -> BuiltInVars:
        raise NotImplementedError()

    def slice(self, loss_slice: slice) -> ISimlVariables:
        raise NotImplementedError()

    def send(self, device: str) -> ISimlVariables:
        raise NotImplementedError()

    def min_len(self) -> int:
        raise NotImplementedError()

    def to_numpy(self) -> Any:
        raise NotImplementedError()


def siml_tensor_variables(values: BuiltInVars) -> ISimlVariables:
    if isinstance(values, torch.Tensor):
        return TensorSimlVariables(values)
    if isinstance(values, dict):
        return DictSimlVariables(values)
    if isinstance(values, list):
        return ListSimlVaraiables(values)

    if isinstance(values, ListSimlVaraiables):
        return ListSimlVaraiables(values.get_values())
    if isinstance(values, DictSimlVariables):
        return DictSimlVariables(values.get_values())
    if isinstance(values, TensorSimlVariables):
        return TensorSimlVariables(values.get_values())

    raise NotImplementedError(
        f"Converter for data type: {type(values)} is not implemented"
    )


class TensorSimlVariables(ISimlVariables):
    def __init__(self, value: torch.Tensor):
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                "cannot initialize objects except torch.Tensor."
                f" Input: {type(value)}"
            )
        self._x = value

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {str(self._x)}"

    def get_values(self) -> torch.Tensor:
        return self._x

    def slice(self, loss_slice: slice) -> TensorSimlVariables:
        tmp = self._x[loss_slice]
        return TensorSimlVariables(tmp)

    def send(self, device: str) -> TensorSimlVariables:
        tmp = self._x.to(device)
        return TensorSimlVariables(tmp)

    def min_len(self) -> int:
        return len(self._x)

    def to_numpy(self) -> Any:
        return self._x.cpu().detach().numpy()


class DictSimlVariables(ISimlVariables):
    def __init__(self, value: dict):
        self._x = {k: siml_tensor_variables(v) for k, v in value.items()}

    def __str__(self) -> str:
        txt = ""
        for k, v in self._x.items():
            txt += f"{k}: {v}, "
        return f"{self.__class__.__name__}" + '{' + txt + '}'

    def get_values(self) -> dict[str, torch.Tensor]:
        tmp = {k: v.get_values() for k, v in self._x.items()}
        return tmp

    def slice(self, loss_slice: slice) -> DictSimlVariables:
        tmp = {k: v.slice(loss_slice) for k, v in self._x.items()}
        return DictSimlVariables(tmp)

    def send(self, device: str) -> TensorSimlVariables:
        tmp = {k: v.send(device) for k, v in self._x.items()}
        return DictSimlVariables(tmp)

    def min_len(self) -> int:
        return np.min([x.min_len() for x in self._x.values()])

    def to_numpy(self) -> Any:
        return {k: v.to_numpy() for k, v in self._x.items()}


class ListSimlVaraiables(ISimlVariables):
    def __init__(self, value: list[torch.Tensor]):
        self._x = [siml_tensor_variables(v) for v in value]

    def __str__(self) -> str:
        values = [str(v) for v in self._x]
        txt = ", ".join(values)
        return f"{self.__class__.__name__}" + '[' + txt + ']'

    def get_values(self) -> list[torch.Tensor]:
        tmp = [v.get_values() for v in self._x]
        return tmp

    def slice(self, loss_slice: slice) -> ListSimlVaraiables:
        tmp = [v.slice(loss_slice) for v in self._x]
        return ListSimlVaraiables(tmp)

    def send(self, device: str) -> ListSimlVaraiables:
        tmp = [v.send(device) for v in self._x]
        return ListSimlVaraiables(tmp)

    def min_len(self) -> int:
        return np.min([x.min_len() for x in self._x])

    def to_numpy(self) -> Any:
        return [v.to_numpy() for v in self._x]

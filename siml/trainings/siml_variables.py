from __future__ import annotations
from typing import TypeVar
import torch


BuiltInVars = TypeVar(
    'BuiltInVars',
    torch.Tensor, dict[torch.Tensor], list[torch.Tensor]
)


class ISimlVaraibles():
    def __init__(self, value: BuiltInVars) -> None:
        raise NotImplementedError()

    def get_value(self) -> BuiltInVars:
        raise NotImplementedError()

    def slice(self, loss_slice: slice) -> ISimlVaraibles:
        raise NotImplementedError()

    def send(self, device: str) -> ISimlVaraibles:
        raise NotImplementedError()


def siml_varaibles(values: BuiltInVars) -> ISimlVaraibles:
    if isinstance(values, torch.Tensor):
        return TensorSimlVariables(values)
    if isinstance(values, dict):
        return DictSimlVariables(values)
    if isinstance(values, list):
        return ListSimlVaraiables(values)

    raise NotImplementedError(
        f"Converter for data type: {type(values)} is not implemented"
    )


class TensorSimlVariables(ISimlVaraibles):
    def __init__(self, value: torch.Tensor):
        self._x = value

    def get_value(self) -> torch.Tensor:
        return self._x

    def slice(self, loss_slice: slice) -> TensorSimlVariables:
        tmp = self._x[loss_slice]
        return TensorSimlVariables(tmp)

    def send(self, device: str) -> TensorSimlVariables:
        tmp = self._x.to(device)
        return TensorSimlVariables(tmp)


class DictSimlVariables(ISimlVaraibles):
    def __init__(self, value: dict[str, torch.Tensor]):
        self._x = {k: TensorSimlVariables(v) for k, v in value.items()}

    def get_value(self) -> dict[str, torch.Tensor]:
        tmp = {k: v.get_value() for k, v in self._x.items()}
        return tmp

    def slice(self, loss_slice: slice) -> DictSimlVariables:
        tmp = {k: v.slice(loss_slice) for k, v in self._x.items()}
        return DictSimlVariables(tmp)

    def send(self, device: str) -> TensorSimlVariables:
        tmp = {k: v.send(device) for k, v in self._x.items()}
        return DictSimlVariables(tmp)


class ListSimlVaraiables(ISimlVaraibles):
    def __init__(self, value: list[torch.Tensor]):
        self._x = [TensorSimlVariables(v) for v in value]

    def get_value(self) -> list[torch.Tensor]:
        tmp = [v.get_value() for v in self._x]
        return tmp

    def slice(self, loss_slice: slice) -> ListSimlVaraiables:
        tmp = [v.slice(loss_slice) for v in self._x]
        return ListSimlVaraiables(tmp)

    def send(self, device: str) -> ListSimlVaraiables:
        tmp = [v.send(device) for v in self._x]
        return ListSimlVaraiables(tmp)

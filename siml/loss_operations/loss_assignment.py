import abc
from typing import Union


class ILossAssignment(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def loss_names(self) -> list[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def __getitem__(self,
                    variable_name: str) -> str:
        raise NotImplementedError()


class LossAssignmentCreator():
    @classmethod
    def create(
        self,
        loss_setting: Union[dict, str]
    ) -> ILossAssignment:
        if type(loss_setting) is dict:
            return DictLossAssignment(loss_setting)

        if type(loss_setting) is str:
            return StrLossAssignment(loss_setting)

        raise NotImplementedError(
            f"Loss Assignment for {type(loss_setting)} is not implemented.")


class DictLossAssignment(ILossAssignment):
    def __init__(self, loss_setting: dict):
        self._loss_setting = loss_setting

    @property
    def loss_names(self) -> list[str]:
        return list(self._loss_setting.values())

    def __getitem__(self, variable_name: str) -> str:
        return self._loss_setting[variable_name]


class StrLossAssignment(ILossAssignment):
    def __init__(self, loss_setting: str):
        self._loss_setting = loss_setting

    @property
    def loss_names(self) -> list[str]:
        return [self._loss_setting]

    def __getitem__(self, variable_name: str) -> str:
        return self._loss_setting

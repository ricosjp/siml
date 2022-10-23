import abc
from typing import List, Union


class ILossAssignment(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def loss_names() -> List[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def __getitem__(self,
                    variable_name: str) -> str:
        raise NotImplementedError()


class LossAssignmentCreator():
    @classmethod
    def create(
            self,
            loss_setting: Union[dict, str]) -> ILossAssignment:
        if type(loss_setting) is dict:
            return DictLossAssignment(loss_setting)

        if type(loss_setting) is str:
            return StrLossAssignment(loss_setting)


class DictLossAssignment(ILossAssignment):
    def __init__(self, loss_setting: dict):
        self.loss_setting = loss_setting

    @property
    def loss_names(self):
        return list(self.loss_setting.values())

    def __getitem__(self,
                    variable_name: str) -> str:
        return self.loss_setting[variable_name]


class StrLossAssignment(ILossAssignment):
    def __init__(self, loss_setting: str):
        self.loss_setting = loss_setting

    @property
    def loss_names(self):
        return self.loss_setting

    def __getitem__(self,
                    variable_name: str) -> str:
        return self.loss_setting

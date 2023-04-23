from __future__ import annotations
import abc

import numpy as np


class ISimlScaler(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def use_diagonal(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def is_erroneous(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def partial_fit(self, data: np.ndarray) -> ISimlScaler:
        raise NotImplementedError()

    @abc.abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

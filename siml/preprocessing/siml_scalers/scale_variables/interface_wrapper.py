import abc
from typing import TypeVar

import numpy as np
import scipy.sparse as sp

T = TypeVar('T', np.ndarray, sp.coo_matrix, sp.csr_matrix, sp.csc_matrix)


class IScalerInputVariables(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, data: T) -> None:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int]:
        raise NotImplementedError()

    @abc.abstractmethod
    def reshape(self, componentwise: bool, **kwrds) -> T:
        raise NotImplementedError()

import abc
from typing import TypeVar, Callable

import numpy as np
import scipy.sparse as sp

T = TypeVar('T', np.ndarray, sp.coo_matrix, sp.csr_matrix, sp.csc_matrix)


class ISimlArray(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, data: T) -> None:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int]:
        raise NotImplementedError()

    @abc.abstractmethod
    def apply(
        self,
        function: Callable[[T], T],
        componentwise: bool,
        *,
        skip_nan: bool = False,
        use_diagonal: bool = False,
        **kwards
    ) -> T:
        """Apply user defined function

        Parameters
        ----------
        function : Callable[[T], T]
            function to apply
        componentwise : bool
            If True, fucnction is applied by component wise way
        skip_nan : bool, optional
            If True, np.nan value is ignored. This option is valid
             only when data is np.ndarray. By default False
        use_diagonal : bool, optional
            If True, only diagonal values are used. This option is valid
             only when data is sparse array. By default False

        Returns
        -------
        T: Same type of instance caraible

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reshape(
        self,
        componentwise: bool,
        *,
        skip_nan: bool = False,
        use_diagonal: bool = False,
        **kwrds
    ) -> T:
        raise NotImplementedError()

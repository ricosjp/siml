from typing import Union
import datetime as dt

import scipy.sparse as sp

from .interface_wrapper import IScalerInputVariables

SparseArrayType = Union[sp.coo_matrix, sp.csr_matrix, sp.csc_matrix]


class SpaceArrayWrapper(IScalerInputVariables):
    def __init__(
        self,
        data: SparseArrayType
    ) -> None:
        self.data = data

    @property
    def shape(self) -> tuple[int]:
        return self.data.shape

    def reshape(
        self,
        componentwise: bool,
        *,
        use_diagonal: bool = False,
        **kwards
    ) -> SparseArrayType:

        if componentwise:
            return self.data

        if use_diagonal:
            print('Start diagonal')
            print(dt.datetime.now())
            reshaped = self.data.diagonal()
            print('Start apply')
            print(dt.datetime.now())
            return reshaped

        reshaped = self.data.reshape(
            (self.shape[0] * self.shape[1], 1)
        )
        return reshaped

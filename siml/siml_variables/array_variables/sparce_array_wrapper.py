import datetime as dt
from typing import Callable
import warnings

from siml.base.siml_typing import SparseArrayType

from .interface_wrapper import ISimlArray


class SparseArrayWrapper(ISimlArray):
    def __init__(
        self,
        data: SparseArrayType
    ) -> None:
        self.data = data

    @property
    def shape(self) -> tuple[int]:
        return self.data.shape

    def apply(
        self,
        apply_function: Callable[[SparseArrayType], SparseArrayType],
        componentwise: bool,
        *,
        use_diagonal: bool = False,
        **kwards
    ) -> SparseArrayType:

        if use_diagonal:
            raise ValueError(
                'Cannot set use_diagonal=True in self.apply function. '
                'use_diagonal is only allows in self.reshape'
            )

        reshaped_array = self.reshape(
            componentwise=componentwise,
            use_diagonal=use_diagonal
        )

        result = apply_function(reshaped_array)
        return result.reshape(self.shape).tocoo()

    def reshape(
        self,
        componentwise: bool,
        *,
        use_diagonal: bool = False,
        **kwards
    ) -> SparseArrayType:
        if componentwise and use_diagonal:
            warnings.warn(
                "component_wise and use_diagonal cannnot use"
                " at the same time. use_diagonal is ignored"
            )

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

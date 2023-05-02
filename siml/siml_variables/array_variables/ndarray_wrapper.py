from typing import Callable

import numpy as np

from .interface_wrapper import ISimlArray


class NdArrayWrapper(ISimlArray):
    def __init__(
        self,
        data: np.ndarray
    ):
        self.data = data

    @property
    def shape(self) -> tuple[int]:
        return self.data.shape

    def apply(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        componentwise: bool,
        *,
        skip_nan: bool = False,
        **kwards
    ) -> np.ndarray:

        reshaped = self.reshape(
            componentwise=componentwise,
            skip_nan=skip_nan
        )
        result = function(reshaped)
        return np.reshape(result, self.shape)

    def reshape(
        self,
        componentwise: bool,
        *,
        skip_nan: bool = False,
        **kwards
    ) -> np.ndarray:

        if componentwise:
            reshaped = np.reshape(
                self.data,
                (np.prod(self.shape[:-1]), self.shape[-1])
            )
        else:
            reshaped = np.reshape(self.data, (-1, 1))

        if not skip_nan:
            return reshaped

        if componentwise:
            isnan = np.isnan(np.prod(reshaped, axis=-1))
            reshaped_wo_nan = reshaped[~isnan]
            return reshaped_wo_nan
        else:
            reshaped_wo_nan = reshaped[~np.isnan(reshaped)][:, None]
            return reshaped_wo_nan

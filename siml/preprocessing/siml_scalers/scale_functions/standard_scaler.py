import numpy as np
from sklearn import preprocessing

from .interface_scaler import ISimlScaler


class StandardScaler(preprocessing.StandardScaler, ISimlScaler):
    def __init__(
        self,
        *,
        copy=True,
        with_mean=True,
        with_std=True,
        **kwargs
    ):
        super().__init__(
            copy=copy,
            with_mean=with_mean,
            with_std=with_std
        )

    @property
    def use_diagonal(self) -> bool:
        return False

    def is_erroneous(self) -> bool:
        return np.any(np.isnan(self.var_))

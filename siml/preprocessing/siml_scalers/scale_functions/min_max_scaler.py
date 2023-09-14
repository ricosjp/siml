from sklearn import preprocessing

from .interface_scaler import ISimlScaler


class MinMaxScaler(preprocessing.MinMaxScaler, ISimlScaler):
    def __init__(
        self,
        feature_range=...,
        *,
        copy=True,
        clip=False,
        **kwargs
    ):
        super().__init__(feature_range, copy=copy, clip=clip)

    @property
    def use_diagonal(self) -> bool:
        return False

    def is_erroneous(self) -> bool:
        return False

from sklearn.base import BaseEstimator, TransformerMixin

from .interface_scaler import ISimlScaler


class IdentityScaler(TransformerMixin, BaseEstimator, ISimlScaler):
    """Class to perform identity conversion (do nothing)."""

    def __init__(self, **kwargs) -> None:
        super().__init__()

    @property
    def use_diagonal(self) -> bool:
        return False

    def is_erroneous(self) -> bool:
        return False

    def partial_fit(self, data):
        return

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

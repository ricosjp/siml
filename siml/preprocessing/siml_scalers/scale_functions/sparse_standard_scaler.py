import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin

from .interface_scaler import ISimlScaler


class SparseStandardScaler(TransformerMixin, BaseEstimator, ISimlScaler):
    """Class to perform standardization for sparse data."""

    def __init__(self, power=1., other_components=None, **kwargs):
        self.var_ = 0.
        self.std_ = 0.
        self.mean_square_ = 0.
        self.n_ = 0
        self.power = power

        if other_components is None:
            other_components = []
        self.component_dim = len(other_components) + 1
        return

    @property
    def use_diagonal(self) -> bool:
        return False

    def is_erroneous(self) -> bool:
        return np.any(np.isnan(self.var_))

    @property
    def vars(self) -> float:
        return self.var_

    def partial_fit(self, data):
        self._raise_if_not_sparse(data)
        self._update(data)
        return self

    def _update(self, sparse_dats):
        m = np.prod(sparse_dats.shape)
        mean_square = (
            self.mean_square_ * self.n_ + np.sum(sparse_dats.data**2)) / (
                self.n_ + m)

        self.mean_square_ = mean_square
        self.n_ += m

        # To use mean_i [x_i^2 + y_i^2 + z_i^2], multiply by the dim
        self.var_ = self.mean_square_ * self.component_dim
        self.std_ = np.sqrt(self.var_)
        return

    def _raise_if_not_sparse(self, data):
        if not sp.issparse(data):
            raise ValueError('Data is not sparse')
        return

    def transform(self, data):
        self._raise_if_not_sparse(data)
        if self.std_ == 0.:
            scale = 0.
        else:
            scale = (1 / self.std_)**self.power
        return data * scale

    def inverse_transform(self, data):
        self._raise_if_not_sparse(data)
        inverse_scale = self.std_**(self.power)
        return data * inverse_scale

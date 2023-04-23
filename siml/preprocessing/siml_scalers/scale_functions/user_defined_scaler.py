import pickle
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin

from .interface_scaler import ISimlScaler


class UserDefinedScaler(TransformerMixin, BaseEstimator, ISimlScaler):

    def __init__(self, pkl_path: Path, **kwargs):
        self.scaler = self._load_pkl(pkl_path)
        return

    @property
    def use_diagonal(self) -> bool:
        return False

    def is_erroneous(self) -> bool:
        return False

    def _load_pkl(self, pkl_path: Path):
        with open(pkl_path, 'rb') as fr:
            scaler = pickle.load(fr)
        return scaler

    def partial_fit(self, data):
        return

    def transform(self, data):
        out = self.scaler.transform(data)
        return out

    def inverse_transform(self, data):
        inverse_out = self.scaler.inverse_transform(data)
        return inverse_out

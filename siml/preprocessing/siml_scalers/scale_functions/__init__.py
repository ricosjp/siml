# flake8: noqa
from functools import singledispatch
from typing import Union

from .identity_scaler import IdentityScaler
from .interface_scaler import ISimlScaler
from .isoam_scaler import IsoAMScaler
from .max_abs_scaler import MaxAbsScaler
from .min_max_scaler import MinMaxScaler
from .sparse_standard_scaler import SparseStandardScaler
from .standard_scaler import StandardScaler
from .user_defined_scaler import UserDefinedScaler

# name to scaler class and default arguments
_name_to_siml_scalers: dict[str, tuple[ISimlScaler, dict]] = {
    "identity": (IdentityScaler, {}),
    "standardize": (StandardScaler, {}),
    "std_scale": (StandardScaler, {"with_mean": False}),
    "sparse_std": (SparseStandardScaler, {"power": 1.0}),
    "isoam_scale": (IsoAMScaler, {}),
    "min_max": (MinMaxScaler, {}),
    "max_abs": (MaxAbsScaler, {"power": 1.0})
}


def create_scaler(scaler_name: str, **kwards) -> ISimlScaler:
    if scaler_name.startswith('user_defined'):
        pkl_path = scaler_name.split("?Path=")[1]
        return UserDefinedScaler(pkl_path)

    if scaler_name not in _name_to_siml_scalers.keys():
        raise ValueError(
            f"Unknown preprocessing method: {scaler_name}")

    cls_object, args = _name_to_siml_scalers[scaler_name]
    args |= kwards
    return cls_object(**kwards)

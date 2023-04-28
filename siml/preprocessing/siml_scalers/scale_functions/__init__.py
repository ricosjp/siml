# flake8: noqa
from .identity_scaler import IdentityScaler
from .interface_scaler import ISimlScaler
from .isoam_scaler import IsoAMScaler
from .max_abs_scaler import MaxAbsScaler
from .min_max_scaler import MinMaxScaler
from .sparse_standard_scaler import SparseStandardScaler
from .standard_scaler import StandardScaler
from .user_defined_scaler import UserDefinedScaler

# name to scaler class and default arguments


def _create_args(scaler_name: str) -> tuple[ISimlScaler, dict]:
    if scaler_name == "identity":
        return (IdentityScaler, {})
    if scaler_name == "standardize":
        return (StandardScaler, {})
    if scaler_name == "std_scale":
        return (StandardScaler, {"with_mean": False})
    if scaler_name == "sparse_std":
        return (SparseStandardScaler, {"power": 1.0})
    if scaler_name == "isoam_scale":
        return (IsoAMScaler, {})
    if scaler_name == "min_max":
        return (MinMaxScaler, {})
    if scaler_name == "max_abs":
        return (MaxAbsScaler, {"power": 1.0})

    raise NotImplementedError(
        f"{scaler_name} is not defined."
    )


def create_scaler(scaler_name: str, **kwards) -> ISimlScaler:
    if scaler_name.startswith('user_defined'):
        pkl_path = scaler_name.split("?Path=")[1]
        return UserDefinedScaler(pkl_path)

    cls_object, args = _create_args(scaler_name)
    args |= kwards
    return cls_object(**args)

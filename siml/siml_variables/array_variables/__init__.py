# flake8: noqa
from typing import get_args

import scipy.sparse as sp
import numpy as np

from siml.base.siml_typing import ArrayDataType

from .interface_wrapper import ISimlArray
from .ndarray_wrapper import NdArrayWrapper
from .sparce_array_wrapper import SparseArrayWrapper, SparseArrayType


def create_siml_arrray(
    data: ArrayDataType
) -> ISimlArray:
    if isinstance(data, np.ndarray):
        return NdArrayWrapper(data)

    if isinstance(data, get_args(SparseArrayType)):
        return SparseArrayWrapper(data)

    raise ValueError(f"Unsupported data type: {data.__class__}")

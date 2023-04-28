# flake8: noqa
from typing import Union

import numpy as np
import scipy.sparse as sp

from .interface_wrapper import ISimlArray
from .ndarray_wrapper import NdArrayWrapper
from .sparce_array_wrapper import SparseArrayWrapper, SparseArrayType

ArrayDataType = Union[
    np.ndarray, sp.coo_matrix, sp.csr_matrix, sp.csc_matrix
]


def create_siml_arrray(
    data: ArrayDataType
) -> ISimlArray:
    if isinstance(data, np.ndarray):
        return NdArrayWrapper(data)

    if isinstance(data, (sp.coo_matrix, sp.csr_matrix, sp.csc_matrix)):
        return SparseArrayWrapper(data)

    raise ValueError(f"Unsupported data type: {data.__class__}")

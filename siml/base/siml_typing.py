from typing import Union

import scipy.sparse as sp
import numpy as np


ArrayDataType = Union[
    np.ndarray, sp.coo_matrix, sp.csr_matrix, sp.csc_matrix
]

SparseArrayType = Union[sp.coo_matrix, sp.csr_matrix, sp.csc_matrix]

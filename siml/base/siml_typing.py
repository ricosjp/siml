from typing import Union

import numpy as np
import scipy.sparse as sp


ArrayDataType = Union[
    np.ndarray, sp.coo_matrix, sp.csr_matrix, sp.csc_matrix
]

SparseArrayType = Union[sp.coo_matrix, sp.csr_matrix, sp.csc_matrix]

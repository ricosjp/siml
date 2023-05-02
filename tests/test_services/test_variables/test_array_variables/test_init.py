import pytest
import numpy as np
import scipy.sparse as sp
from typing import get_args

from siml.siml_variables import create_siml_arrray
from siml.siml_variables.array_variables \
    import NdArrayWrapper, SparseArrayType, SparseArrayWrapper


@pytest.mark.parametrize("value", [
    ([1., 2., 3.1]),
    ("aaaaa")
])
def test__cannot_initialize(value):
    with pytest.raises(ValueError):
        _ = create_siml_arrray(value)


@pytest.mark.parametrize("value", [
    (np.random.rand(10, 3)),
    (sp.csr_matrix(np.random.rand(10, 3))),
    (sp.coo_matrix(np.random.rand(10, 3))),
    (sp.csc_matrix(np.random.rand(10, 3)))
])
def test__can_intialize(value):
    result = create_siml_arrray(value)

    if isinstance(value, np.ndarray):
        assert isinstance(result, NdArrayWrapper)
        return

    elif isinstance(value, get_args(SparseArrayType)):
        assert isinstance(result, SparseArrayWrapper)
        return

    pytest.fail(
        f"type: {type(value)} is not undertood."
    )

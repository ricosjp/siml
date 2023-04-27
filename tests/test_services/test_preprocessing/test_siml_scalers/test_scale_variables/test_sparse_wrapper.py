import numpy as np
import pytest
import scipy.sparse as sp

from siml.preprocessing.siml_scalers.scale_variables import SparseArrayWrapper


@pytest.mark.parametrize("value", [
    (sp.csr_matrix(np.random.rand(10, 3))),
    (sp.coo_matrix(np.random.rand(10, 3))),
    (sp.csc_matrix(np.random.rand(10, 3)))
])
def test__shape(value):
    wrapper = SparseArrayWrapper(value)
    assert wrapper.shape == value.shape


@pytest.mark.parametrize("value, expect_shape", [
    (sp.csr_matrix(np.random.rand(10, 3)), (10, 3)),
    (sp.coo_matrix(np.random.rand(20, 5)), (20, 5)),
    (sp.csc_matrix(np.random.rand(11, 4)), (11, 4))
])
def test__component_wise_reshape(value, expect_shape):
    wrapper = SparseArrayWrapper(value)
    result = wrapper.reshape(componentwise=True)
    assert result.shape == expect_shape


@pytest.mark.parametrize("value, expect_shape", [
    (sp.csr_matrix(np.random.rand(10, 3)), (30, 1)),
    (sp.coo_matrix(np.random.rand(20, 5)), (100, 1)),
    (sp.csc_matrix(np.random.rand(11, 4)), (44, 1))
])
def test__not_component_wise_reshape(value, expect_shape):
    wrapper = SparseArrayWrapper(value)
    result = wrapper.reshape(componentwise=False)
    assert result.shape == expect_shape


@pytest.mark.parametrize("value, expect_shape", [
    (sp.csr_matrix(np.random.rand(10, 3)), (3,)),
    (sp.coo_matrix(np.random.rand(20, 5)), (5,)),
    (sp.csc_matrix(np.random.rand(11, 4)), (4,))
])
def test__diagonal_reshape(value, expect_shape):
    wrapper = SparseArrayWrapper(value)
    result = wrapper.reshape(componentwise=False, use_diagonal=True)
    assert result.shape == expect_shape


@pytest.mark.parametrize("value, component_wise", [
    (sp.csr_matrix(np.random.rand(10, 3)), True),
    (sp.coo_matrix(np.random.rand(20, 5)), False),
    (sp.csc_matrix(np.random.rand(11, 4)), False)
])
def test__apply_not_use_diagonal(value, component_wise):
    func = lambda x: 2 * x  # noqa: E731

    wrapper = SparseArrayWrapper(value)
    result = wrapper.apply(
        func,
        componentwise=component_wise,
        use_diagonal=False
    )

    assert result.shape == value.shape
    assert isinstance(result, sp.coo_matrix)

    np.testing.assert_array_almost_equal(
        result.todense(),
        func(value).todense()
    )


@pytest.mark.parametrize("value, component_wise", [
    (sp.csr_matrix(np.random.rand(10, 3)), True),
    (sp.coo_matrix(np.random.rand(20, 5)), False),
    (sp.csc_matrix(np.random.rand(11, 4)), False),
])
def test__apply_use_diagonal(value, component_wise):
    func = lambda x: x * 2  # noqa: E731

    wrapper = SparseArrayWrapper(value)
    with pytest.raises(ValueError):
        _ = wrapper.apply(
            func,
            componentwise=component_wise,
            use_diagonal=True
        )

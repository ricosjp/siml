import numpy as np
import pytest

from siml.siml_variables.array_variables import NdArrayWrapper


@pytest.mark.parametrize("value", [
    (np.random.rand(10, 3))
])
def test__shape(value):
    wrapper = NdArrayWrapper(value)
    assert wrapper.shape == value.shape


@pytest.mark.parametrize("value, expect_shape", [
    (np.random.rand(10, 3), (10, 3)),
    (np.random.rand(10, 4, 3), (40, 3))
])
def test___component_wise_reshape(value, expect_shape):
    wrapper = NdArrayWrapper(value)
    reshaped = wrapper.reshape(componentwise=True)

    assert reshaped.shape == expect_shape


@pytest.mark.parametrize("value, expect_shape", [
    (np.random.rand(10, 3), (30, 1)),
    (np.random.rand(10, 4, 3), (120, 1))
])
def test___not_component_wise_reshape(value, expect_shape):
    wrapper = NdArrayWrapper(value)
    reshaped = wrapper.reshape(componentwise=False)

    assert reshaped.shape == expect_shape


@pytest.mark.parametrize("value, expect_shape", [
    (np.random.rand(10, 3), (10, 3)),
    (np.random.rand(10, 4, 3), (40, 3))
])
def test__include_nan_reshaped(value, expect_shape):
    random_pos = np.random.choice([True, False], value.shape)
    value[random_pos] = np.nan
    assert np.any(np.isnan(value))

    wrapper = NdArrayWrapper(value)
    reshaped = wrapper.reshape(
        componentwise=True,
        skip_nan=False
    )

    assert np.any(np.isnan(reshaped))
    assert reshaped.shape == expect_shape


@pytest.mark.parametrize("value, expect_shape", [
    (np.random.rand(10, 3), (30, 1)),
    (np.random.rand(10, 4, 3), (120, 1))
])
def test__skip_nan_reshaped_not_component_wise(value, expect_shape):
    random_pos = np.random.choice([True, False], value.shape)
    value[random_pos] = np.nan
    assert np.any(np.isnan(value))

    wrapper = NdArrayWrapper(value)
    reshaped = wrapper.reshape(
        componentwise=False,
        skip_nan=True
    )

    assert np.all(~np.isnan(reshaped))

    n_not_nan = np.sum(~random_pos)
    assert reshaped.shape == (n_not_nan, 1)


@pytest.mark.parametrize("value, component_wise", [
    (np.random.rand(10, 3), True),
    (np.random.rand(10, 4, 3), True),
    (np.random.rand(10, 3), False),
    (np.random.rand(10, 4, 3), False),
])
def test__apply(value, component_wise):
    func = lambda x: x * 2  # noqa: E731

    wrapper = NdArrayWrapper(value)
    result = wrapper.apply(
        func,
        componentwise=component_wise
    )
    assert result.shape == value.shape
    np.testing.assert_array_almost_equal(result, func(value))

# @pytest.mark.parametrize("value, expect_shape", [
#     (np.random.rand(10, 3), (10, 3)),
#     (np.random.rand(10, 4, 3), (40, 3))
# ])
# def test__apply_component_wise(value, expect_shape):
#     func = lambda x: x * 2  # noqa: E731

#     wrapper = NdArrayWrapper(value)
#     result = wrapper.apply(
#         func,
#         componentwise=True
#     )
#     assert result.shape == value.shape
#     assert result == func(value)

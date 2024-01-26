import pytest

from siml.networks.cross_product import CrossProduct
from siml import setting
import numpy as np
import torch


@pytest.mark.parametrize("x0, x1", [
    (
        np.array([1, 0, 0]),
        np.array([0, 1, 0])
    ),
    (
        np.array([1, 2, 2]),
        np.array([1, 2, 2])
    ),
    (
        np.random.rand(3),
        np.random.rand(3)
    )
])
def test__cross_product_for_single_vector(x0, x1):
    model = CrossProduct(setting.BlockSetting(type="cross_product"))

    t_x0 = torch.tensor(x0.reshape(-1, 1))
    t_x1 = torch.tensor(x1.reshape(-1, 1))
    actual: torch.Tensor = model.forward(t_x0, t_x1)

    desired = np.cross(x0, x1).reshape(-1, 1)

    np.testing.assert_almost_equal(
        actual=actual.detach().numpy(),
        desired=desired
    )


@pytest.mark.parametrize("x0, x1", [
    (
        torch.randn(10, 3, 1),
        torch.randn(10, 3, 1)
    )
])
def test__cross_product_for_single_vector_in_multiple_nodes(
    x0: torch.Tensor, x1: torch.Tensor
):
    model = CrossProduct(setting.BlockSetting(type="cross_product"))

    actual: torch.Tensor = model.forward(x0, x1)
    actual_array = actual.detach().numpy()

    # greedy check
    n_node = x0.shape[0]
    for i in range(n_node):
        x0i = x0[i].detach().numpy()
        x1i = x1[i].detach().numpy()
        desired = np.cross(x0i, x1i, axis=0)

        np.testing.assert_almost_equal(
            actual=actual_array[i],
            desired=desired
        )


@pytest.mark.parametrize("x0, x1", [
    (
        torch.randn(10, 3, 5),
        torch.randn(10, 3, 5)
    )
])
def test__cross_product_for_single_vector_with_features_in_multiple_nodes(
    x0: torch.Tensor, x1: torch.Tensor
):
    model = CrossProduct(setting.BlockSetting(type="cross_product"))

    actual: torch.Tensor = model.forward(x0, x1)
    actual_array = actual.detach().numpy()

    # greedy check
    n_node = x0.shape[0]
    n_feature = x0.shape[-1]
    for i in range(n_node):
        for j in range(n_feature):
            x0i = x0[i, :, j].detach().numpy()
            x1i = x1[i, :, j].detach().numpy()
            desired = np.cross(x0i, x1i, axis=0)

            np.testing.assert_almost_equal(
                actual=actual_array[i, :, j],
                desired=desired
            )


@pytest.mark.parametrize("x0, x1", [
    (
        torch.randn(5, 10, 3, 24),
        torch.randn(5, 10, 3, 24),
    )
])
def test__cross_product_for_timeseries_vector_with_features_in_multiple_nodes(
    x0: torch.Tensor, x1: torch.Tensor
):
    model = CrossProduct(setting.BlockSetting(type="cross_product"))

    actual: torch.Tensor = model.forward(x0, x1)
    actual_array = actual.detach().numpy()

    # greedy check
    n_times = x0.shape[0]
    n_node = x0.shape[1]
    n_feature = x0.shape[-1]
    for t in range(n_times):
        for i in range(n_node):
            for j in range(n_feature):
                x0i = x0[t, i, :, j].detach().numpy()
                x1i = x1[t, i, :, j].detach().numpy()
                desired = np.cross(x0i, x1i, axis=0)

                np.testing.assert_almost_equal(
                    actual=actual_array[t, i, :, j],
                    desired=desired
                )


@pytest.mark.parametrize("x0, x1", [
    (
        torch.randn(5, 10, 3, 24),
        torch.randn(10, 3, 24),
    )
])
def test__cross_product_for_oneside_timeseries_vector(
    x0: torch.Tensor, x1: torch.Tensor
):
    model = CrossProduct(setting.BlockSetting(type="cross_product"))

    actual: torch.Tensor = model.forward(x0, x1)
    actual_array = actual.detach().numpy()

    # greedy check
    n_times = x0.shape[0]
    n_node = x0.shape[1]
    n_feature = x0.shape[-1]
    for t in range(n_times):
        for i in range(n_node):
            for j in range(n_feature):
                x0i = x0[t, i, :, j].detach().numpy()
                x1i = x1[i, :, j].detach().numpy()
                desired = np.cross(x0i, x1i, axis=0)

                np.testing.assert_almost_equal(
                    actual=actual_array[t, i, :, j],
                    desired=desired
                )

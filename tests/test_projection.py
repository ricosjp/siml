import pytest

from siml.networks.projection import Projection
from siml import setting
import numpy as np
import torch


@pytest.fixture
def mean_projection() -> Projection:
    block_setting = setting.BlockSetting(
        type='projection',
        optional={"operator": 'mean'}
    )
    return Projection(block_setting)


@pytest.fixture
def time_series_mean_projection() -> Projection:
    block_setting = setting.BlockSetting(
        type='projection',
        optional={"operator": 'mean',
                  'time_series_input': True}
    )
    return Projection(block_setting)


@pytest.fixture
def sum_projection() -> Projection:
    block_setting = setting.BlockSetting(
        type='projection',
        optional={"operator": 'sum'}
    )
    return Projection(block_setting)


@pytest.fixture
def time_series_sum_projection() -> Projection:
    block_setting = setting.BlockSetting(
        type='projection',
        optional={"operator": 'sum',
                  'time_series_input': True}
    )
    return Projection(block_setting)


@pytest.fixture
def setup_input_tensor():
    inputs = []

    # Suppose shape of input tensor is (n_vertex, dim)
    # ex. scalar values
    flag = torch.tensor([1, 1, 1, -1, -1, 0, 0])
    value_tensor = torch.tensor([3, 4, 2, 5, 5, 6, 6], dtype=float)
    mean_tensor = torch.tensor([3, 4, 2, 3, 3, 6, 6], dtype=float)
    sum_tensor = torch.tensor([3, 4, 2, 9, 9, 6, 6], dtype=float)
    inputs.append((flag, value_tensor, mean_tensor, sum_tensor))
    for _tensor in inputs[0]:
        _tensor = torch.unsqueeze(_tensor, -1)

    # Suppose shape of input tensor is (n_vertex, dim, dim)
    # ex. U (Velocity)
    flag = torch.tensor([1, 1, 1, -1, -1, 0, 0])
    value_tensor = torch.tensor([[1, 1, 1], [1, 3, 3], [1, 2, 5],
                                 [3, 4, 5], [5, 4, 2], [2, 2, 2],
                                 [3, 3, 3]], dtype=float)
    mean_tensor = torch.tensor([[1, 1, 1], [1, 3, 3], [1, 2, 5],
                                [1, 2, 3], [1, 2, 3], [2, 2, 2],
                                [3, 3, 3]], dtype=float)
    sum_tensor = torch.tensor([[1, 1, 1], [1, 3, 3], [1, 2, 5],
                               [3, 6, 9], [3, 6, 9], [2, 2, 2],
                               [3, 3, 3]], dtype=float)
    inputs.append((flag, value_tensor, mean_tensor, sum_tensor))
    for _tensor in inputs[1]:
        _tensor = torch.unsqueeze(_tensor, -1)

    return inputs


@pytest.fixture
def setup_time_series_input_scalar_tensor():
    inputs = []
    # Suppose shape of input tensor is (n_vertex, dim)
    # ex. scalar values
    flag = torch.tensor([1, 1, 1, -1, -1, 0, 0]).unsqueeze(-1)
    value_tensor = torch.tensor([[3, 4, 2, 5, 5, 6, 6],
                                 [4, 4, 7, 5, 8, 9, 9]],
                                dtype=float).unsqueeze(-1)
    mean_tensor = torch.tensor([[3, 4, 2, 3, 3, 6, 6],
                                [4, 4, 7, 5, 5, 9, 9]],
                               dtype=float).unsqueeze(-1)
    sum_tensor = torch.tensor([[3, 4, 2, 9, 9, 6, 6],
                               [4, 4, 7, 15, 15, 9, 9]],
                              dtype=float).unsqueeze(-1)
    inputs.append((flag, value_tensor, mean_tensor, sum_tensor))

    # Suppose shape of input tensor is (n_vertex, dim, dim)
    # ex. U (Velocity)
    flag = torch.tensor([1, 1, 1, -1, -1, 0, 0])
    value_tensor = torch.tensor([[[1, 1, 1], [1, 3, 3], [1, 2, 5],
                                 [3, 4, 5], [5, 4, 2], [2, 2, 2],
                                 [3, 3, 3]],
                                 [[1, 2, 0], [5, 3, 4], [0, 4, 8],
                                 [3, 4, 5], [5, 4, 2], [2, 2, 2],
                                 [3, 3, 3]]
                                 ], dtype=float)
    mean_tensor = torch.tensor([[[1, 1, 1], [1, 3, 3], [1, 2, 5],
                                [1, 2, 3], [1, 2, 3], [2, 2, 2],
                                [3, 3, 3]],
                                [[1, 2, 0], [5, 3, 4], [0, 4, 8],
                                [2, 3, 4], [2, 3, 4], [2, 2, 2],
                                [3, 3, 3]]], dtype=float)
    sum_tensor = torch.tensor([[[1, 1, 1], [1, 3, 3], [1, 2, 5],
                               [3, 6, 9], [3, 6, 9], [2, 2, 2],
                               [3, 3, 3]],
                               [[1, 2, 0], [5, 3, 4], [0, 4, 8],
                               [6, 9, 12], [6, 9, 12], [2, 2, 2],
                               [3, 3, 3]]], dtype=float)
    inputs.append((flag, value_tensor, mean_tensor, sum_tensor))
    return inputs


def test__default_setting():
    block_setting = setting.BlockSetting(
        type='projection'
    )
    block = Projection(block_setting)
    assert block.op == torch.mean


def test__mean_forward(mean_projection, setup_input_tensor):
    model: Projection = mean_projection
    inputs = setup_input_tensor
    for flag, value_tensor, mean_tensor, _ in inputs:
        pred = model.forward(value_tensor, flag)
        np.testing.assert_array_almost_equal(mean_tensor, pred)


def test__time_series_mean_forward(time_series_mean_projection,
                                   setup_time_series_input_scalar_tensor):
    model: Projection = time_series_mean_projection
    inputs = setup_time_series_input_scalar_tensor
    for flag, value_tensor, mean_tensor, _ in inputs:
        pred = model.forward(value_tensor, flag)
        np.testing.assert_array_almost_equal(mean_tensor, pred)


def test__time_series_sum_forward(time_series_sum_projection,
                                  setup_time_series_input_scalar_tensor):
    model: Projection = time_series_sum_projection
    inputs = setup_time_series_input_scalar_tensor
    for flag, value_tensor, _, sum_tensor in inputs:
        pred = model.forward(value_tensor, flag)
        np.testing.assert_array_almost_equal(sum_tensor, pred)


def test__sum_forward(sum_projection, setup_input_tensor):
    model: Projection = sum_projection
    inputs = setup_input_tensor
    for flag, value_tensor, _, sum_tensor in inputs:
        pred = model.forward(value_tensor, flag)
        np.testing.assert_array_almost_equal(sum_tensor, pred)

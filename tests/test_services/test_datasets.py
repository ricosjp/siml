import pathlib
import unittest

import numpy as np
import pytest
import torch

import siml.datasets as datasets


class TestDatasets(unittest.TestCase):

    def test_merge_sparse_tensors_square(self):
        stripped_sparse_info = [
            {
                'size': [2, 2],
                'row': torch.Tensor([0, 1, 1]),
                'col': torch.Tensor([0, 0, 1]),
                'values': torch.Tensor([1., 2., 3.]),
            },
            {
                'size': [2, 2],
                'row': torch.Tensor([0, 1, 1]),
                'col': torch.Tensor([0, 0, 1]),
                'values': torch.Tensor([10., 20., 30.]),
            },
            {
                'size': [2, 2],
                'row': torch.Tensor([0, 1, 1]),
                'col': torch.Tensor([0, 0, 1]),
                'values': torch.Tensor([100., 200., 300.]),
            },
        ]
        expected_sparse = np.array([
            [1., 0., 0., 0., 0., 0.],
            [2., 3., 0., 0., 0., 0.],
            [0., 0., 10., 0., 0., 0.],
            [0., 0., 20., 30., 0., 0.],
            [0., 0., 0., 0., 100., 0.],
            [0., 0., 0., 0., 200., 300.],
        ])
        merged_sparse = datasets.merge_sparse_tensors(stripped_sparse_info)
        np.testing.assert_almost_equal(
            merged_sparse.to_dense().numpy(), expected_sparse)

    def test_merge_sparse_tensors_rectangle(self):
        stripped_sparse_info = [
            {
                'size': [2, 5],
                'row': torch.Tensor([0, 1, 1, 1]),
                'col': torch.Tensor([0, 0, 1, 4]),
                'values': torch.Tensor([1., 2., 3., 4.]),
            },
            {
                'size': [3, 4],
                'row': torch.Tensor([0, 1, 1, 2]),
                'col': torch.Tensor([0, 0, 1, 3]),
                'values': torch.Tensor([10., 20., 30., 40.]),
            },
            {
                'size': [4, 2],
                'row': torch.Tensor([0, 1, 1, 3]),
                'col': torch.Tensor([0, 0, 1, 1]),
                'values': torch.Tensor([100., 200., 300., 400.]),
            },
        ]
        expected_sparse = np.array([
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [2., 3., 0., 0., 4., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 10., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 20., 30., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 40., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 100., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 200., 300.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 400.],
        ])
        merged_sparse = datasets.merge_sparse_tensors(stripped_sparse_info)
        np.testing.assert_almost_equal(
            merged_sparse.to_dense().numpy(), expected_sparse)


def test__simplified_dataset_not_initialized():
    x_variable_names = ["x1", "x2"]
    y_variable_names = ["y"]
    raw_dict_x = {
        "x1": torch.tensor(np.random.rand(3, 4)),
        "x2": torch.tensor(np.random.rand(3, 4))
    }
    with pytest.raises(ValueError):
        _ = datasets.SimplifiedDataset(
            x_variable_names=x_variable_names,
            y_variable_names=y_variable_names,
            raw_dict_x=raw_dict_x,
            directories=[
                pathlib.Path("path/to/raw"),
                pathlib.Path("path/to/raw")
            ]
        )


def test__simplified_dataset_len():
    x_variable_names = ["x1", "x2"]
    y_variable_names = ["y"]
    raw_dict_x = {
        "x1": torch.tensor(np.random.rand(3, 4)),
        "x2": torch.tensor(np.random.rand(3, 4))
    }
    dataset = datasets.SimplifiedDataset(
        x_variable_names=x_variable_names,
        y_variable_names=y_variable_names,
        raw_dict_x=raw_dict_x
    )

    assert len(dataset) == 1


def test__simplified_dataset_get_item_without_support():
    x_variable_names = ["x1", "x2"]
    y_variable_names = ["y"]
    raw_dict_x = {
        "x1": np.random.rand(10, 3),
        "x2": np.random.rand(10, 5)
    }
    dataset = datasets.SimplifiedDataset(
        x_variable_names=x_variable_names,
        y_variable_names=y_variable_names,
        raw_dict_x=raw_dict_x
    )

    expect = np.concatenate([raw_dict_x["x1"], raw_dict_x["x2"]], axis=-1)
    actual = dataset[0]
    np.testing.assert_almost_equal(
        actual=actual['x'].numpy(), desired=expect
    )
    assert actual['t'] is None
    assert actual['data_directory'] is None
    assert 'supports' not in actual


def test__simplified_dataset_get_item_with_support():
    x_variable_names = ["x2"]
    y_variable_names = ["y"]
    raw_dict_x = {
        "x1": np.random.rand(10, 3),
        "x2": np.random.rand(10, 5)
    }
    supports = ["x1"]
    dataset = datasets.SimplifiedDataset(
        x_variable_names=x_variable_names,
        y_variable_names=y_variable_names,
        raw_dict_x=raw_dict_x,
        supports=supports
    )

    actual = dataset[0]
    np.testing.assert_almost_equal(
        actual=actual['x'].numpy(), desired=raw_dict_x["x2"]
    )
    assert actual['t'] is None
    assert actual['data_directory'] is None
    assert 'supports' in actual

    actual_supports = actual["supports"]
    np.testing.assert_almost_equal(
        actual_supports[0], raw_dict_x["x1"]
    )

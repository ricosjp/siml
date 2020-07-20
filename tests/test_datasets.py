import unittest

import numpy as np
import torch

import siml.datasets as datasets


class TestDatasets(unittest.TestCase):

    def test_merge_sparse_tensors(self):
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

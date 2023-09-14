
import unittest

import numpy as np
from scipy.stats import ortho_group
import torch

import siml.networks.tcn as tcn
import siml.setting as setting


class TestTCN(unittest.TestCase):

    def _test_equivariance(self, layer, ts_rank1):
        ts_rank1_h = layer(ts_rank1)
        np.testing.assert_array_equal(
            ts_rank1_h.shape[:-1], ts_rank1.shape[:-1])
        ortho_mat = torch.from_numpy(ortho_group.rvs(3).astype(np.float32))

        rotated_ts_rank1 = torch.einsum('pq,tnqf->tnpf', ortho_mat, ts_rank1)
        rotated_ts_rank1_h = torch.einsum(
            'pq,tnqf->tnpf', ortho_mat, ts_rank1_h)
        actual = layer(rotated_ts_rank1)
        np.testing.assert_almost_equal(
            actual.detach().numpy(), rotated_ts_rank1_h.detach().numpy(),
            decimal=3)
        return

    def test_linear_tcn(self):
        layer = tcn.TCN(
            setting.BlockSetting(
                nodes=[4, 2], activations=['identity'],
                bias=False,
                kernel_sizes=[2],
            ))
        ts_rank1_1 = np.reshape(np.arange(
            10 * 8 * 3 * 4), (10, 8, 3, 4))  # (time, node, dim, feature)
        ts_rank1_2 = np.reshape(np.arange(
            100 * 2 * 3 * 4), (100, 2, 3, 4))  # (time, node, dim, feature)

        self._test_equivariance(
            layer, torch.from_numpy(ts_rank1_1.astype(np.float32)))
        self._test_equivariance(
            layer, torch.from_numpy(ts_rank1_2.astype(np.float32)))
        return

    def test_equivariant_tcn(self):
        layer = tcn.EquivariantTCN(
            setting.BlockSetting(
                nodes=[4, 4, 4], activations=['tanh', 'tanh'],
                bias=True,
                kernel_sizes=[2, 2],
            ))
        ts_rank1_1 = np.reshape(np.arange(
            10 * 8 * 3 * 4), (10, 8, 3, 4))  # (time, node, dim, feature)
        ts_rank1_2 = np.reshape(np.arange(
            100 * 2 * 3 * 4), (100, 2, 3, 4))  # (time, node, dim, feature)

        self._test_equivariance(
            layer, torch.from_numpy(ts_rank1_1.astype(np.float32)))
        self._test_equivariance(
            layer, torch.from_numpy(ts_rank1_2.astype(np.float32)))
        return

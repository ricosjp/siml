
import unittest

import numpy as np
import torch

import siml.networks.einops as einops
import siml.setting as setting


class TestEinops(unittest.TestCase):

    def test_rearrange(self):
        layer = einops.Rearrange(
            setting.BlockSetting(optional={
                'pattern': 'time node feature -> (time feature) node'
            }))
        array = np.reshape(np.arange(
            10 * 3 * 5), (10, 3, 5))  # (time, node, feature)

        actual = layer(torch.from_numpy(array))
        desired = np.concatenate([
            a for a in np.swapaxes(array, 1, 2)], axis=0)
        np.testing.assert_almost_equal(actual, desired)

    def test_rearrange_axes_lengths(self):
        layer = einops.Rearrange(
            setting.BlockSetting(optional={
                'pattern': 'time node (f1 f2) -> (time f1) node f2',
                'axes_lengths': {'f1': 3},
            }))
        array = np.reshape(np.arange(
            10 * 5 * 6), (10, 5, 6))  # (time, node, feature)
        reshaped_array = np.swapaxes(np.reshape(array, (10, 5, 3, 2)), 1, 2)

        actual = layer(torch.from_numpy(array))
        desired = np.concatenate([
            a for a in reshaped_array], axis=0)
        np.testing.assert_almost_equal(actual, desired)

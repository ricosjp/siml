import pathlib
import shutil
import unittest

import numpy as np
import torch

import siml.networks.nan_mlp as nan_mlp
import siml.setting as setting
import siml.trainer as trainer


class TestNaNMLP(unittest.TestCase):

    def test_nan_mlp(self):
        pad_value = 1.
        nan_mlp_net = nan_mlp.NaNMLP(
            setting.BlockSetting(
                nodes=[2, 4, 8], activations=['tanh', 'identity'],
                optional={'pad_value': pad_value}))
        x = np.random.rand(10, 2).astype(np.float32)
        x[2, 0] = np.nan
        x[-1] = np.nan
        y = np.random.rand(10, 8).astype(np.float32)
        tensor_pred = nan_mlp_net(torch.from_numpy(x))
        loss = torch.mean((tensor_pred - torch.from_numpy(y))**2)
        self.assertFalse(torch.any(torch.isnan(loss)))

        loss.backward()
        for linear in nan_mlp_net.linears:
            self.assertFalse(torch.any(torch.isnan(linear.weight.grad)))
            self.assertFalse(torch.any(torch.isnan(linear.bias.grad)))

        pred = tensor_pred.detach().numpy()
        np.testing.assert_almost_equal(pred[2], pad_value)
        np.testing.assert_almost_equal(pred[-1], pad_value)

    def test_nan_mlp_time_series(self):
        pad_value = 0.
        nan_mlp_net = nan_mlp.NaNMLP(
            setting.BlockSetting(
                nodes=[2, 4, 8], activations=['tanh', 'identity'],
                optional={'pad_value': pad_value, 'axis': 1}))
        x = np.random.rand(20, 10, 2).astype(np.float32)
        x[:, 2, 0] = np.nan
        x[:, -1] = np.nan
        y = np.random.rand(20, 10, 8).astype(np.float32)
        tensor_pred = nan_mlp_net(torch.from_numpy(x))
        loss = torch.mean((tensor_pred - torch.from_numpy(y))**2)
        self.assertFalse(torch.any(torch.isnan(loss)))

        loss.backward()
        for linear in nan_mlp_net.linears:
            self.assertFalse(torch.any(torch.isnan(linear.weight.grad)))
            self.assertFalse(torch.any(torch.isnan(linear.bias.grad)))

        pred = tensor_pred.detach().numpy()
        np.testing.assert_almost_equal(pred[:, 2], pad_value)
        np.testing.assert_almost_equal(pred[:, -1], pad_value)

    def test_train_nan_mlp(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            pathlib.Path('tests/data/data_with_nan/model.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, .1)

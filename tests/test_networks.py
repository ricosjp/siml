from pathlib import Path
import shutil
import unittest

import numpy as np
import matplotlib.pyplot as plt
import torch

import siml.networks.header as header
import siml.setting as setting
import siml.trainer as trainer


PLOT = False


class TestNetwork(unittest.TestCase):

    def test_deepsets(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/deepsets.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)

    def test_deepsets_permutation(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/deepsets.yml'))
        tr = trainer.Trainer(main_setting)
        tr.prepare_training()
        x = np.reshape(np.arange(5*3), (1, 5, 3)).astype(np.float32) * .1

        y_wo_permutation = tr.model({'x': torch.from_numpy(x)})

        x_w_permutation = np.concatenate(
            [x[0, None, 2:], x[0, None, :2]], axis=1)
        y_w_permutation = tr.model({'x': torch.from_numpy(x_w_permutation)})

        np.testing.assert_almost_equal(
            np.concatenate(
                [
                    y_wo_permutation[0, None, 2:].detach().numpy(),
                    y_wo_permutation[0, None, :2].detach().numpy()],
                axis=1),
            y_w_permutation.detach().numpy())

    def test_res_gcn(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/res_gcn.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_gcn(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/gcn.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_nri(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/nri.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_nri_non_concat(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/nri.yml'))
        main_setting.model.blocks[0].optional['concat'] = False
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_reduce(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/reduce.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_reduce_mlp(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/reduce_mlp.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_deform_gradient(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/res_gcn_grad.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_deform_gradient_share(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/res_gcn_grad.yml'))
        main_setting.model.blocks[0].optional['multiple_networks'] = False
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_train_time_series_simplified_data(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/simplified_timeseries/lstm.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, .1)

    def test_train_time_series_mesh_data_w_support(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform_timeseries/lstm_w_support.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1.)

    def test_train_time_series_mesh_data_wo_support(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform_timeseries/lstm_wo_support.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1.)

    def test_train_res_ltm(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform_timeseries/res_lstm.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1.)

    def test_train_tcn(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform_timeseries/tcn.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1.)

    def test_activations(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/activations.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1.)

    def test_mish(self):
        np.testing.assert_almost_equal(
            header.mish(torch.Tensor([100.])), [100.])
        np.testing.assert_almost_equal(
            header.mish(torch.Tensor([-100.])), [0.])
        np.testing.assert_almost_equal(
            header.mish(torch.Tensor([1.])),
            [1. * np.tanh(np.log(1 + np.exp(1.)))])
        if PLOT:
            x = np.linspace(-10., 10., 100)
            mish = header.mish(torch.from_numpy(x))
            plt.plot(x, mish.numpy())
            plt.show()

    def test_no_bias(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/no_bias.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)
        self.assertIsNone(tr.model.chains[0].linears[0].bias)

    def test_time_norm(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform_timeseries/time_norm.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1.)
        input_data = tr.train_loader.dataset[0]
        input_data = {'x': input_data['x'][:, None, :, :]}
        out = tr.model(input_data)
        np.testing.assert_almost_equal(out.detach().numpy()[0], 0.)

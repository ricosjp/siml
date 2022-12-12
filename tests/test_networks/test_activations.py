from pathlib import Path
import shutil
import unittest

import numpy as np
import matplotlib.pyplot as plt
import torch

import siml.networks.activations as activations
import siml.setting as setting
import siml.trainer as trainer


PLOT = False


class TestActivations(unittest.TestCase):

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
            activations.mish(torch.Tensor([100.])), [100.])
        np.testing.assert_almost_equal(
            activations.mish(torch.Tensor([-100.])), [0.])
        np.testing.assert_almost_equal(
            activations.mish(torch.Tensor([1.])),
            [1. * np.tanh(np.log(1 + np.exp(1.)))])
        if PLOT:
            x = np.linspace(-10., 10., 100)
            mish = activations.mish(torch.from_numpy(x))
            plt.plot(x, mish.numpy())
            plt.show()

    def test_normalize(self):
        a = np.random.rand(10, 3)
        normalized_a = activations.normalize(torch.from_numpy(a)).numpy()
        np.testing.assert_almost_equal(
            normalized_a, a / (np.linalg.norm(a, axis=1)[..., None] + 1e-5))

    def test_inverse_leaky_relu(self):
        leaky_relu = torch.nn.LeakyReLU()
        inversed_leaky_relu = activations.InversedLeakyReLU(leaky_relu)

        tensor = torch.rand(100, 3)
        tensor_ = inversed_leaky_relu(leaky_relu(tensor))

        np.testing.assert_almost_equal(
            tensor_.detach().numpy(), tensor.numpy())

    def test_derivative_leaky_relu(self):
        tensor = torch.linspace(-1, 0, 1000)
        leaky_relu = torch.nn.LeakyReLU()
        derivative_leaky_relu = activations.DerivativeLeakyReLU(leaky_relu)
        h_derivative = derivative_leaky_relu(tensor)
        np.testing.assert_almost_equal(
            h_derivative.detach().numpy()[:-1], leaky_relu.negative_slope)

        tensor = torch.linspace(0, 1, 1000)
        leaky_relu = torch.nn.LeakyReLU()
        derivative_leaky_relu = activations.DerivativeLeakyReLU(leaky_relu)
        h_derivative = derivative_leaky_relu(tensor)
        np.testing.assert_almost_equal(
            h_derivative.detach().numpy()[1:], 1.)
        np.testing.assert_almost_equal(
            h_derivative.detach().numpy()[0],
            (1 + leaky_relu.negative_slope) / 2)

    def test_inverse_tanh(self):
        tensor = torch.rand(100, 3)
        tensor_ = activations.atanh(torch.tanh(tensor))

        np.testing.assert_almost_equal(
            tensor_.detach().numpy(), tensor.numpy())

    def test_smooth_leaky_relu(self):
        tensor = torch.rand(100, 3)
        tensor_ = activations.inversed_smooth_leaky_relu(
            activations.smooth_leaky_relu(tensor))
        np.testing.assert_almost_equal(
            tensor_.detach().numpy(), tensor.numpy(), decimal=5)

    def test_smooth_leaky_relu_extreme(self):
        x_large = torch.tensor([1e5])
        x_small = torch.tensor([-1e5])

        np.testing.assert_almost_equal(
            activations.smooth_leaky_relu(x_large).detach().numpy(), 1e5)
        np.testing.assert_almost_equal(
            activations.smooth_leaky_relu(x_small).detach().numpy(), - 1e5 / 2)

        np.testing.assert_almost_equal(
            activations.inversed_smooth_leaky_relu(
                x_large).detach().numpy(), 1e5)
        np.testing.assert_almost_equal(
            activations.inversed_smooth_leaky_relu(
                x_small).detach().numpy(), - 1e5 * 2)

        np.testing.assert_almost_equal(
            activations.derivative_smooth_leaky_relu(
                x_large).detach().numpy(), 1.)
        np.testing.assert_almost_equal(
            activations.derivative_smooth_leaky_relu(
                x_small).detach().numpy(), .5)

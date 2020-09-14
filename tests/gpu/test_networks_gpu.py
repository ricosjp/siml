from pathlib import Path
import shutil
import unittest

from chainer import testing
import matplotlib.pyplot as plt
import numpy as np

import siml.inferer as inferer
import siml.setting as setting
import siml.trainer as trainer


PLOT = False


class TestNetworksGPU(unittest.TestCase):

    @testing.attr.multi_gpu(2)
    def test_iso_gcn_gpu(self):
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress/'
            'iso_gcn_dict_input_dict_output.yml'))
        main_setting.trainer.gpu_id = 1
        main_setting.trainer.num_workers = 0  # Serial
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    @testing.attr.multi_gpu(2)
    def test_nri_gpu(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/nri.yml'))
        main_setting.trainer.gpu_id = 1
        main_setting.trainer.num_workers = 0  # Serial
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    @testing.attr.multi_gpu(2)
    def test_res_gcn_gpu(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/res_gcn.yml'))
        main_setting.trainer.gpu_id = 1
        main_setting.trainer.num_workers = 0  # Serial
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 5.)

    @testing.attr.multi_gpu(2)
    def test_integration_y0(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/ode/integration_y0.yml'))
        main_setting.trainer.gpu_id = 1

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1e-2)

        ir = inferer.Inferer(main_setting)
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            preprocessed_data_directory=main_setting.data.preprocessed_root
            / 'test',
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        self.assertLess(results[0]['loss'], 1e-1)
        if PLOT:
            cmap = plt.get_cmap('tab10')
            for i, result in enumerate(results):
                plt.plot(
                    result['dict_x']['t'][:, 0, 0],
                    result['dict_x']['y0'][:, 0, 0],
                    ':', color=cmap(i), label=f"y0 answer of data {i}")
                plt.plot(
                    result['dict_x']['t'][:, 0, 0],
                    result['dict_y']['y0'][:, 0, 0],
                    color=cmap(i), label=f"y0 inferred of data {i}")
            plt.legend()
            plt.show()

    @testing.attr.multi_gpu(2)
    def test_integration_y1(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/ode/integration_y1.yml'))
        main_setting.trainer.gpu_id = 1

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 5e-2)

        ir = inferer.Inferer.read_settings(
            main_setting.trainer.output_directory / 'settings.yml')
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            preprocessed_data_directory=main_setting.data.preprocessed_root
            / 'test',
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        self.assertLess(results[0]['loss'], 1e-1)
        if PLOT:
            cmap = plt.get_cmap('tab10')
            for i, result in enumerate(results):
                plt.plot(
                    result['dict_x']['t'][:, 0, 0],
                    result['dict_x']['y1'][:, 0, 0],
                    ':', color=cmap(i), label=f"y1 answer of data {i}")
                plt.plot(
                    result['dict_x']['t'][:, 0, 0],
                    result['dict_y']['y1'][:, 0, 0],
                    color=cmap(i), label=f"y1 inferred of data {i}")
            plt.legend()
            plt.show()

    @testing.attr.multi_gpu(2)
    def test_integration_y2(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/ode/integration_y2.yml'))
        main_setting.trainer.gpu_id = 1

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 5e-1)

        ir = inferer.Inferer(main_setting)
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            preprocessed_data_directory=main_setting.data.preprocessed_root
            / 'test',
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        self.assertLess(results[0]['loss'], .2)
        if PLOT:
            cmap = plt.get_cmap('tab10')
            for i, result in enumerate(results):
                plt.plot(
                    result['dict_x']['t'][:, 0, 0],
                    result['dict_x']['y2'][:, 0, 0],
                    ':', color=cmap(i), label=f"y2 answer of data {i}")
                plt.plot(
                    result['dict_x']['t'][:, 0, 0],
                    result['dict_y']['y2'][:, 0, 0],
                    color=cmap(i), label=f"y2 inferred of data {i}")
            plt.legend()
            plt.show()

    @testing.attr.multi_gpu(2)
    def test_integration_y3(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/ode/integration_y3.yml'))
        main_setting.trainer.gpu_id = 1

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 2e-1)

        ir = inferer.Inferer(main_setting)
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            preprocessed_data_directory=main_setting.data.preprocessed_root
            / 'test',
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        self.assertLess(results[0]['loss'], 5e-1)
        if PLOT:
            cmap = plt.get_cmap('tab10')
            for i, result in enumerate(results):
                plt.plot(
                    result['dict_x']['t'][:, 0, 0],
                    result['dict_x']['y3'][:, 0, 0],
                    ':', color=cmap(i), label=f"y3 answer of data {i}")
                plt.plot(
                    result['dict_x']['t'][:, 0, 0],
                    result['dict_y']['y3'][:, 0, 0],
                    color=cmap(i), label=f"y3 inferred of data {i}")
            plt.legend()
            plt.show()

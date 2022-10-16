from pathlib import Path
import shutil
import sys

import numpy as np
import torch

import siml.inferer as inferer
import siml.setting as setting
import siml.trainer as trainer


sys.path.insert(0, 'tests')
import equivariance_base  # NOQA


if torch.cuda.is_available():
    GPU_ID = 0
else:
    GPU_ID = -1


class TestDGGNN(equivariance_base.EquivarianceBase):

    def validate(self, main_setting, model_directory, decimal=1e-7):
        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=model_directory,
            data_directories=main_setting.data.validation)
        result = results[0]
        volume = np.load(result['data_directory'] / 'cell_volume.npy')
        self.evaluate_conservation(
            result['dict_x']['cell_initial_phi'],
            result['dict_y']['cell_phi'],
            volume=volume, decimal=decimal,
            target_time_series=False, prediction_time_series=True)

        transformed_path = main_setting.data.validation[0].parent \
            / f"transformed_{main_setting.data.validation[0].name}"
        if not transformed_path.exists():
            return

        transformed_results = ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=model_directory,
            data_directories=transformed_path)
        transformed_result = transformed_results[0]
        transformed_volume = np.load(
            transformed_result['data_directory'] / 'cell_volume.npy')
        self.evaluate_conservation(
            transformed_result['dict_x']['cell_initial_phi'],
            transformed_result['dict_y']['cell_phi'],
            volume=transformed_volume, decimal=decimal,
            target_time_series=False, prediction_time_series=True)
        # np.testing.assert_almost_equal(
        #     transformed_result['loss'], result['loss'])
        np.testing.assert_almost_equal(
            transformed_result['dict_y']['cell_phi'],
            result['dict_y']['cell_phi'], decimal=decimal)
        return

    def test_simplest_dggnn(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/advection/simplest_dggnn.yml'))
        main_setting.trainer.gpu_id = GPU_ID
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

        self.validate(main_setting, tr.setting.trainer.output_directory)

    def test_linear_dggnn(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/advection/linear_dggnn.yml'))
        main_setting.trainer.gpu_id = GPU_ID
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

        self.validate(main_setting, tr.setting.trainer.output_directory)

    def test_nonlinear_dggnn(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/advection/nonlinear_dggnn.yml'))
        main_setting.trainer.gpu_id = GPU_ID
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

        self.validate(main_setting, tr.setting.trainer.output_directory)

    def test_freeshape_simplest_dggnn(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/freeshape_advection/simplest_dggnn.yml'))
        main_setting.trainer.gpu_id = GPU_ID
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

        self.validate(
            main_setting, tr.setting.trainer.output_directory, decimal=1e-6)

    def test_freeshape_linear_dggnn(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/freeshape_advection/linear_dggnn.yml'))
        main_setting.trainer.gpu_id = GPU_ID
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

        self.validate(
            main_setting, tr.setting.trainer.output_directory, decimal=1e-4)

    def test_freeshape_nonlinear_dggnn(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/freeshape_advection/nonlinear_dggnn.yml'))
        main_setting.trainer.gpu_id = GPU_ID
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

        self.validate(main_setting, tr.setting.trainer.output_directory)

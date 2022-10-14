from pathlib import Path
import shutil
import unittest

import numpy as np
import torch

import siml.inferer as inferer
import siml.setting as setting
import siml.trainer as trainer


if torch.cuda.is_available():
    GPU_ID = 0
else:
    GPU_ID = -1


class TestNetworks(unittest.TestCase):

    def evaluate_conservation(
            self, target, prediction, *,
            volume=None,
            target_time_series=False, prediction_time_series=False,
            decimal=4):

        if volume is None:
            if target_time_series:
                volume = np.ones(len(target[0]))[..., None]
            else:
                volume = np.ones(len(target))[..., None]

        if target_time_series:
            target_conservation = np.sum(target[0] * volume)
        else:
            target_conservation = np.sum(target * volume)

        if prediction_time_series:
            prediction_conservation = np.einsum(
                'ti...,i->t...', prediction, volume[..., 0])
        else:
            prediction_conservation = np.einsum(
                'i...,i->...', prediction, volume[..., 0])
        np.testing.assert_almost_equal(
            prediction_conservation - target_conservation, 0., decimal=decimal)
        print(prediction_conservation - target_conservation)
        return

    def test_dggnn(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/advection/dggnn.yml'))
        main_setting.trainer.gpu_id = GPU_ID
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=main_setting.data.validation)
        result = results[0]
        self.evaluate_conservation(
            result['dict_x']['cell_initial_phi'],
            result['dict_y']['cell_phi'],
            target_time_series=False, prediction_time_series=True)

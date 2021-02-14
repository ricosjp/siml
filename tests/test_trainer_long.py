from pathlib import Path
import shutil
import unittest

import numpy as np
# import torch

import siml.setting as setting
import siml.trainer as trainer


PLOT = False


class TestTrainerLong(unittest.TestCase):

    def test_train_cpu(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear.yml'))
        main_setting.trainer.num_workers = 0  # Serial
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1e-5)

    def test_train_general_block(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/mlp_long.yml'))
        main_setting.trainer.num_workers = 0  # Serial
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1e-2)

    def test_train_res_gcn(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/res_gcn_long.yml'))
        main_setting.trainer.num_workers = 0  # Serial
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1e-2)

    # def test_train_lstm_long(self):
    #     main_setting = setting.MainSetting.read_settings_yaml(
    #         Path('tests/data/deform_timeseries/lstm_long.yml'))
    #     tr = trainer.Trainer(main_setting)
    #     if tr.setting.trainer.output_directory.exists():
    #         shutil.rmtree(tr.setting.trainer.output_directory)
    #     loss = tr.train()
    #     np.testing.assert_array_less(loss, 1e-2)
    #
    #     preprocessed_data_directory = Path(
    #         'tests/data/deform_timeseries/preprocessed/train'
    #         '/tet2_3_modulusx1.0000')
    #     x = np.concatenate([
    #         np.load(preprocessed_data_directory / 't.npy'),
    #         np.load(preprocessed_data_directory / 'strain.npy'),
    #         np.load(preprocessed_data_directory / 'modulus.npy'),
    #     ], axis=-1)[:, None, :, :]
    #     y = np.load(preprocessed_data_directory / 'stress.npy')
    #     tr.model.eval()
    #     with torch.no_grad():
    #         inferred_y = tr.model({'x': torch.from_numpy(x)}).numpy()
    #     eval = np.mean((y - inferred_y[:, 0, :, :])**2)
    #     self.assertLess(eval, 1e-2)

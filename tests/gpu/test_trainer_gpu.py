from pathlib import Path
import shutil
import unittest

from chainer import testing
import numpy as np

import siml.setting as setting
import siml.trainer as trainer


class TestTrainerGPU(unittest.TestCase):

    @testing.attr.multi_gpu(2)
    def test_train_general_block_gpu(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/general_block.yml'))
        main_setting.trainer.gpu_id = 1
        main_setting.trainer.num_workers = 0  # Serial
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 3.)

    @testing.attr.multi_gpu(2)
    def test_train_gpu(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear.yml'))
        main_setting.trainer.gpu_id = 1
        main_setting.trainer.num_workers = 0  # Serial
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1e-5)

    @testing.attr.multi_gpu(2)
    def test_train_data_parallel(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear.yml'))
        main_setting.trainer.data_parallel = True
        main_setting.trainer.num_workers = 0  # Serial
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    @testing.attr.multi_gpu(2)
    def test_train_data_parallel_general_block(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/general_block.yml'))
        main_setting.trainer.data_parallel = True
        main_setting.trainer.num_workers = 0  # Serial
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 2.)

    @testing.attr.multi_gpu(2)
    def test_train_model_parallel_general_block(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/general_block_model_parallel.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 2.)

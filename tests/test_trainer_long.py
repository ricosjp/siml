import shutil
import unittest

from chainer import testing
import numpy as np

import siml.setting as setting
import siml.trainer as trainer


class TestTrainer(unittest.TestCase):

    def test_train_cpu(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/linear/linear.yml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1e-5)

    def test_train_general_block(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/deform/mlp_long.yml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1e-2)

    @testing.attr.multi_gpu(2)
    def test_train_gpu(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/linear/linear.yml')
        main_setting.trainer.gpu_id = 1
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1e-5)

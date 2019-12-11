from pathlib import Path
import shutil
import unittest

import numpy as np

import siml.setting as setting
import siml.trainer as trainer


class TestTrainer(unittest.TestCase):

    def test_train_cpu(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1e-5)

    def test_train_general_block(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/mlp_long.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 2e-2)

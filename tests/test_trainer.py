import shutil
import unittest

import numpy as np

import siml.setting as setting
import siml.trainer as trainer


class TestTrainer(unittest.TestCase):

    def test_train_cpu_short(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/linear/linear_short.yml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)

    def test_train_general_block_without_support(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/deform/general_block_wo_support.yml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_train_general_block(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/deform/general_block.yml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_train_general_block_input_selection(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/deform/general_block_input_selection.yml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

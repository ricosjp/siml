from pathlib import Path
import shutil
import unittest

import numpy as np

import siml.setting as setting
import siml.trainer as trainer


class TestGroups(unittest.TestCase):

    def test_group_simple(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation_thermal_stress/group_simple.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_group_repeat(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation_thermal_stress/group_repeat.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

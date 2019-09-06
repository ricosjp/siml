from pathlib import Path
import unittest

import numpy as np

import siml.setting as setting


class TestSetting(unittest.TestCase):

    def test_write_yaml(self):
        main_setting = setting.MainSetting()
        write_setting_yml = Path('tests/data/write_setting.yml')
        if write_setting_yml.exists():
            write_setting_yml.unlink()
        setting.write_yaml(main_setting, write_setting_yml)
        written_setting = setting.MainSetting.read_settings_yaml(
            write_setting_yml)
        np.testing.assert_array_equal(
            main_setting.trainer.inputs, written_setting.trainer.inputs)

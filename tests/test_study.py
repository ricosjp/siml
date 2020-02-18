from pathlib import Path
import shutil
import unittest

import numpy as np

import siml.setting as setting
import siml.study as study


class TestTrainer(unittest.TestCase):

    def test_study(self):
        settings_yaml = Path('tests/data/large/study.yml')
        original_setting = setting.MainSetting.read_settings_yaml(
            settings_yaml)
        if original_setting.study.root_directory.exists():
            shutil.rmtree(original_setting.study.root_directory)
        st = study.Study(settings_yaml)
        st.run()

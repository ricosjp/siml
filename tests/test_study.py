from pathlib import Path
import shutil
import unittest

import pandas as pd

import siml.setting as setting
import siml.study as study


class TestTrainer(unittest.TestCase):

    def test_study_long(self):
        settings_yaml = Path('tests/data/large/study_long.yml')
        original_setting = setting.MainSetting.read_settings_yaml(
            settings_yaml)
        if original_setting.study.root_directory.exists():
            shutil.rmtree(original_setting.study.root_directory)
        st = study.Study(settings_yaml)
        st.run()

    def test_study_distributed(self):
        settings_yaml = Path('tests/data/large/study.yml')
        original_setting = setting.MainSetting.read_settings_yaml(
            settings_yaml)
        if original_setting.study.root_directory.exists():
            shutil.rmtree(original_setting.study.root_directory)

        st1 = study.Study(settings_yaml)
        st2 = study.Study(settings_yaml)
        df = pd.read_csv(st1.log_file_path, header=0)
        st1.run_single(df.iloc[0])
        st2.run_single(df.iloc[1])
        st1.run_single(df.iloc[2])
        df = pd.read_csv(st1.log_file_path, header=0)
        conditions = df[df.status != study.Status('FINISHED').value]
        self.assertEqual(len(conditions), 3)

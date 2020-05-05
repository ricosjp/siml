from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np

import siml.optimize as optimize
import siml.setting as setting


class TestOptimize(unittest.TestCase):

    def test_generate_dict(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/optuna.yml'))
        objective = optimize.Objective(main_setting, None)

        dict_replace_1 = {
            'inputs': [{'name': 'abc', 'dim': 6}],
            'n_node': 35,
            'hidden_layers': 11,
            'dropout': 0.01}
        replaced_setting_1 = objective._generate_dict(
            main_setting.optuna.setting, dict_replace_1)

        dict_replace_2 = {
            'inputs': [
                {'name': 'elemental_strain', 'dim': 6},
                {'name': 'something', 'dim': 100}],
            'n_node': 135,
            'hidden_layers': 111,
            'dropout': 0.11}
        replaced_setting_2 = objective._generate_dict(
            main_setting.optuna.setting, dict_replace_2)

        self.assertEqual(
            replaced_setting_1['trainer']['inputs'][0]['name'],
            'abc')
        self.assertEqual(
            replaced_setting_2['trainer']['inputs'][0]['name'],
            'elemental_strain')
        self.assertEqual(
            replaced_setting_2['trainer']['inputs'][1]['name'],
            'something')
        self.assertEqual(
            replaced_setting_2['model']['blocks'][0]['hidden_nodes'], 135)
        self.assertEqual(
            replaced_setting_2['model']['blocks'][0]['hidden_layers'], 111)
        self.assertEqual(
            replaced_setting_2['model']['blocks'][0]['hidden_dropout'], 0.11)

    def test_perform_study(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/optuna.yml'))
        if main_setting.optuna.output_base_directory.exists():
            shutil.rmtree(main_setting.optuna.output_base_directory)
        study = optimize.Study(main_setting)
        study.perform_study()
        self.assertLess(
            study.study.best_trial.value,
            np.max([t.value for t in study.study.trials]))

    def test_perform_study_step_by_step(self):
        main_setting_yml = Path('tests/data/deform/optuna.yml')
        main_setting = setting.MainSetting.read_settings_yaml(
            main_setting_yml)
        if main_setting.optuna.output_base_directory.exists():
            shutil.rmtree(main_setting.optuna.output_base_directory)

        if Path('.venv').exists():
            poetry = 'poetry'
        else:
            poetry = 'python3.7 -m poetry'
        for _ in range(3):
            subprocess.run(
                f"{poetry} run optimize {main_setting_yml} "
                '-s true -l true',
                shell=True, check=True)

        db_setting = setting.DBSetting(use_sqlite=True)
        study = optimize.Study(main_setting, db_setting)
        self.assertEqual(len(study.study.get_trials()), 3)

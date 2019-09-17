from pathlib import Path
import shutil
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
        study = optimize.perform_study(main_setting)
        self.assertLess(
            study.best_trial.value, np.max([t.value for t in study.trials]))

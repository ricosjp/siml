from pathlib import Path
import unittest

import numpy as np

import siml.setting as setting
from siml.path_like_objects import SimlFileBuilder


class TestSetting(unittest.TestCase):

    def test_write_yaml(self):
        main_setting = setting.MainSetting()
        main_setting.misc['string'] = 'abc'
        main_setting.misc['data'] = 1
        main_setting.misc['list'] = [1, 2, 3]
        main_setting.misc['dict'] = {'a': 1, 'b': 2, 'c': 3}

        write_setting_yml = Path('tests/data/write_setting.yml')
        if write_setting_yml.exists():
            write_setting_yml.unlink()
        setting.write_yaml(main_setting, write_setting_yml)
        written_setting = setting.MainSetting.read_settings_yaml(
            write_setting_yml)
        np.testing.assert_array_equal(
            main_setting.trainer.inputs, written_setting.trainer.inputs)
        self.assertEqual(
            main_setting.misc['string'], written_setting.misc['string'])
        self.assertEqual(
            main_setting.misc['data'], written_setting.misc['data'])
        self.assertEqual(
            main_setting.misc['list'], written_setting.misc['list'])
        self.assertEqual(
            main_setting.misc['dict'], written_setting.misc['dict'])

    def test_read_settings_yaml(self):
        yaml_file = Path('tests/data/deform/general_block.yml')
        real_file_setting = setting.MainSetting.read_settings_yaml(yaml_file)

        siml_file = SimlFileBuilder.yaml_file(yaml_file)
        setting_by_file = setting.MainSetting.read_dict_settings(
            siml_file.load()
        )
        self.assertEqual(
            real_file_setting.model.blocks[1].destinations,
            setting_by_file.model.blocks[1].destinations
        )

    def test_read_settings_yaml_data(self):
        yaml_file = Path('tests/data/deform/data.yml')
        main_setting = setting.MainSetting.read_settings_yaml(yaml_file)
        self.assertEqual(
            main_setting.data.interim, [Path('tests/data/deform/interim')])

    def test_main_setting(self):
        main_setting = setting.MainSetting()
        np.testing.assert_array_equal(
            main_setting.conversion.required_file_names, [])
        self.assertEqual(main_setting.data.interim, [Path('data/interim')])

    def test_update_with_dict(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/optuna.yml'))
        dict_replace = {
            'trainer': {
                'inputs': [
                    {'name': 'elemental_strain', 'dim': 6},
                    {'name': 'something', 'dim': 100}],
            }
        }
        new_setting = main_setting.update_with_dict(dict_replace)
        self.assertEqual(
            new_setting.trainer.inputs[0].name, 'elemental_strain')
        self.assertEqual(
            new_setting.trainer.inputs[0].dim, 6)
        self.assertEqual(
            new_setting.trainer.inputs[1].name, 'something')
        self.assertEqual(
            new_setting.trainer.inputs[1].dim, 100)


def test__default_optimizer_setting():
    trainer_setting = setting.TrainerSetting()
    assert trainer_setting.optimizer_setting['lr'] == 0.001
    assert trainer_setting.optimizer_setting['betas'] == (0.9, 0.999)
    assert trainer_setting.optimizer_setting['eps'] == 1e-8
    assert trainer_setting.optimizer_setting['weight_decay'] == 0


def test__enable_to_get_lr():
    trainer_setting = setting.TrainerSetting(
        optimizer_setting={'betas': (0.1, 0.9)}
    )
    assert trainer_setting.optimizer_setting['lr'] == 0.001
    assert trainer_setting.optimizer_setting['betas'] == (0.1, 0.9)

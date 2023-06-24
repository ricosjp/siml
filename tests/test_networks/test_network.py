from pathlib import Path
import shutil
import unittest

import numpy as np
import torch

import siml.networks as networks
import siml.setting as setting
import siml.trainer as trainer


class TestNetwork(unittest.TestCase):

    def test_raise_valueerror_when_network_is_not_dag(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/not_dag.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        with self.assertRaisesRegex(ValueError, 'Cycle found in the network'):
            tr = trainer.Trainer(main_setting)
            tr.train()

    def test_raise_valueerror_when_block_has_no_predecessors(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/no_predecessors.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        with self.assertRaisesRegex(
                ValueError, 'NO_PREDECESSORS has no predecessors'):
            tr = trainer.Trainer(main_setting)
            tr.train()

    def test_raise_valueerror_when_block_has_no_successors(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/no_successors.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        with self.assertRaisesRegex(
                ValueError, 'NO_SUCCESSORS has no successors'):
            tr = trainer.Trainer(main_setting)
            tr.train()

    def test_raise_valueerror_when_block_has_missing_destinations(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/missing_destinations.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        with self.assertRaisesRegex(
                ValueError, 'NOT_EXISTING_BLOCK does not exist'):
            tr = trainer.Trainer(main_setting)
            tr.train()

    def test_node_number_inference(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/node_number_inference.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1e-1)

    def test_user_defined_block(self):

        class CutOffBlock(networks.SimlModule):
            @staticmethod
            def get_name():
                return 'cutoff'

            @staticmethod
            def is_trainable():
                return True

            @staticmethod
            def accepts_multiple_inputs():
                return False

            @staticmethod
            def uses_support():
                return False

            def __init__(self, block_setting):
                super().__init__(block_setting)
                self.upper = block_setting.optional.get('upper', .1)
                self.lower = block_setting.optional.get('lower', 0.)

            def _forward_core(self, x, supports=None, original_shapes=None):
                h = x
                for linear, dropout_ratio, activation in zip(
                        self.linears, self.dropout_ratios, self.activations):
                    h = linear(h)
                    h = activation(h)
                    h[h > self.upper] = self.upper
                    h[h < self.lower] = self.lower
                return h

        networks.add_block(
            block=CutOffBlock)

        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/cutoff.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)
        x = torch.from_numpy(np.random.rand(2000, 100).astype(np.float32))

        out_cutoff1 = tr.model.dict_block['CUTOFF1'](x).detach().numpy()
        np.testing.assert_almost_equal(
            np.max(out_cutoff1),
            main_setting.model.blocks[1].optional['upper'])
        np.testing.assert_almost_equal(
            np.min(out_cutoff1),
            main_setting.model.blocks[1].optional['lower'])

        out_cutoff2 = tr.model.dict_block['CUTOFF2'](x).detach().numpy()
        np.testing.assert_almost_equal(
            np.max(out_cutoff2),
            main_setting.model.blocks[2].optional['upper'])
        np.testing.assert_almost_equal(
            np.min(out_cutoff2),
            main_setting.model.blocks[2].optional['lower'])

    def test_single_dict_output(self):
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress/'
            'iso_gcn_single_dict_output.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 10.)

    def test_dict_output(self):
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress/'
            'iso_gcn_dict_input_dict_output.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1.)

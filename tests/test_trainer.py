import shutil
import unittest

from chainer import testing
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

    @testing.attr.multi_gpu(2)
    def test_train_general_block_gpu(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/deform/general_block.yml')
        main_setting.trainer.gpu_id = 1
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_train_element_wise(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/linear/linear_element_wise.yml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)

    def test_train_element_batch(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/linear/linear_element_batch.yml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)

    def test_siml_updater_equivalent(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/linear/linear_element_batch.yml')

        main_setting.trainer.element_batchsize = 1
        eb1_tr = trainer.Trainer(main_setting)
        if eb1_tr.setting.trainer.output_directory.exists():
            shutil.rmtree(eb1_tr.setting.trainer.output_directory)
        eb1_loss = eb1_tr.train()

        main_setting.trainer.element_batchsize = -1
        ebneg_tr = trainer.Trainer(main_setting)
        if ebneg_tr.setting.trainer.output_directory.exists():
            shutil.rmtree(ebneg_tr.setting.trainer.output_directory)
        ebneg_loss = ebneg_tr.train()

        main_setting.trainer.use_siml_updater = False
        std_tr = trainer.Trainer(main_setting)
        if std_tr.setting.trainer.output_directory.exists():
            shutil.rmtree(std_tr.setting.trainer.output_directory)
        std_loss = std_tr.train()

        np.testing.assert_almost_equal(eb1_loss, std_loss)
        np.testing.assert_almost_equal(ebneg_loss, std_loss)

    def test_train_element_learning_rate(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/linear/linear_short_lr.yml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)

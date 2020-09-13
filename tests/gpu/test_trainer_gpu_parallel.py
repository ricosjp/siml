from pathlib import Path
import shutil
import unittest

from chainer import testing
import numpy as np

import siml.setting as setting
import siml.trainer as trainer


class TestTrainerGPU(unittest.TestCase):

    @testing.attr.multi_gpu(2)
    def test_train_data_parallel_linear(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear.yml'))
        main_setting.data.train = [Path('tests/data/linear/preprocessed')]
        main_setting.trainer.gpu_id = 0
        main_setting.trainer.n_epoch = 1000
        tr_wo_parallel = trainer.Trainer(main_setting)
        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        loss_wo_parallel = tr_wo_parallel.train()

        main_setting.trainer.data_parallel = True
        main_setting.trainer.num_workers = 0  # Serial
        tr_w_parallel = trainer.Trainer(main_setting)
        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        loss_w_parallel = tr_w_parallel.train()
        np.testing.assert_almost_equal(
            loss_w_parallel / loss_wo_parallel, 1., decimal=5)

    @testing.attr.multi_gpu(2)
    def test_train_data_parallel_general_block(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/general_block.yml'))
        main_setting.trainer.num_workers = 0  # Serial

        main_setting.trainer.gpu_id = 0
        tr_wo_parallel = trainer.Trainer(main_setting)
        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        loss_wo_parallel = tr_wo_parallel.train()

        main_setting.trainer.data_parallel = True
        tr_w_parallel = trainer.Trainer(main_setting)
        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        loss_w_parallel = tr_w_parallel.train()
        np.testing.assert_almost_equal(loss_w_parallel / loss_wo_parallel, 1.)

    @testing.attr.multi_gpu(2)
    def test_train_data_parallel_dict_input_dict_output(self):
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress'
            '/iso_gcn_dict_input_dict_output.yml'))
        main_setting.trainer.num_workers = 0  # Serial
        main_setting.trainer.n_epoch = 10
        main_setting.trainer.batch_size = 6

        main_setting.trainer.gpu_id = 0
        tr_wo_parallel = trainer.Trainer(main_setting)
        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        loss_wo_parallel = tr_wo_parallel.train()

        main_setting.trainer.data_parallel = True
        tr_w_parallel = trainer.Trainer(main_setting)
        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        loss_w_parallel = tr_w_parallel.train()
        np.testing.assert_almost_equal(
            loss_w_parallel / loss_wo_parallel, 1., decimal=5)

    def test_train_data_parallel_dict_input_list_output(self):
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress/iso_gcn_dict_input.yml'))
        main_setting.trainer.num_workers = 0  # Serial
        main_setting.trainer.n_epoch = 10
        main_setting.trainer.batch_size = 6

        main_setting.trainer.gpu_id = 0
        tr_wo_parallel = trainer.Trainer(main_setting)
        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        loss_wo_parallel = tr_wo_parallel.train()

        main_setting.trainer.data_parallel = True
        tr_w_parallel = trainer.Trainer(main_setting)
        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        loss_w_parallel = tr_w_parallel.train()
        np.testing.assert_almost_equal(
            loss_w_parallel / loss_wo_parallel, 1., decimal=5)

    @testing.attr.multi_gpu(2)
    def test_train_model_parallel_general_block(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/general_block_model_parallel.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 2.)

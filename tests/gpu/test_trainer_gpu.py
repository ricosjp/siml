from pathlib import Path
import shutil
import unittest

from chainer import testing
import numpy as np

import siml.setting as setting
import siml.trainer as trainer
import siml.util as util


class TestTrainerGPU(unittest.TestCase):

    @testing.attr.multi_gpu(2)
    def test_train_general_block_gpu(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/general_block.yml'))
        main_setting.trainer.gpu_id = 1
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train().cpu()
        np.testing.assert_array_less(loss, 3.)

    @testing.attr.multi_gpu(2)
    def test_train_gpu(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear.yml'))
        main_setting.trainer.gpu_id = 1
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train().cpu()
        np.testing.assert_array_less(loss, 1e-5)

    @testing.attr.multi_gpu(2)
    def test_mulprocess_dataloader_faster(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/large/mlp.yml'))
        main_setting.trainer.gpu_id = 1

        # Generate data
        n_feat = 1000
        n_element = 10000
        n_train_data = 20
        output_root = main_setting.data.train[0]
        if not output_root.exists():
            for i in range(n_train_data):
                x = np.random.rand(n_element, n_feat)
                y = x * 2.
                output_directory = output_root / f"{i}"
                output_directory.mkdir(parents=True)
                np.save(output_directory / 'x.npy', x.astype(np.float32))
                np.save(output_directory / 'y.npy', y.astype(np.float32))

        n_validation_data = 2
        output_root = main_setting.data.validation[0]
        if not output_root.exists():
            for i in range(n_validation_data):
                x = np.random.rand(n_element, n_feat)
                y = x * 2.
                output_directory = output_root / f"{i}"
                output_directory.mkdir(parents=True)
                np.save(output_directory / 'x.npy', x.astype(np.float32))
                np.save(output_directory / 'y.npy', y.astype(np.float32))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        main_setting.trainer.num_workers = 0  # Serial
        tr_workers_0 = trainer.Trainer(main_setting)
        tr_workers_0.train()

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        main_setting.trainer.num_workers = util.determine_max_process()
        tr_workers_max = trainer.Trainer(main_setting)
        tr_workers_max.train()

        # Confirm multiprocess is faster
        raise ValueError()

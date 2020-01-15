from pathlib import Path
import shutil
import unittest

from chainer import testing
import pandas as pd
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
        loss = tr.train()
        np.testing.assert_array_less(loss, 3.)

    @testing.attr.multi_gpu(2)
    def test_train_gpu(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear.yml'))
        main_setting.trainer.gpu_id = 1
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1e-5)

    @testing.attr.multi_gpu(2)
    def test_mulprocess_dataloader_faster(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/large/mlp.yml'))
        main_setting.trainer.gpu_id = 1
        main_setting.trainer.num_workers = 0  # Serial

        # Generate data
        n_feat = 100
        n_element = 20000

        n_train_data = 50
        output_root = main_setting.data.train[0]  # pylint: disable=E1136
        if output_root.exists():
            shutil.rmtree(output_root)
        for i in range(n_train_data):
            x = np.random.rand(n_element, n_feat)
            y = x * 2.
            output_directory = output_root / f"{i}"
            output_directory.mkdir(parents=True)
            np.save(output_directory / 'x.npy', x.astype(np.float32))
            np.save(output_directory / 'y.npy', y.astype(np.float32))

        n_validation_data = 2
        output_root = main_setting.data.validation[0]  # pylint: disable=E1136
        if output_root.exists():
            shutil.rmtree(output_root)
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
        df_workers_0 = pd.read_csv(
            main_setting.trainer.output_directory / 'log.csv', header=0,
            index_col=None, skipinitialspace=True)

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        main_setting.trainer.num_workers = util.determine_max_process(4)
        tr_workers_4 = trainer.Trainer(main_setting)
        tr_workers_4.train()
        df_workers_4 = pd.read_csv(
            main_setting.trainer.output_directory / 'log.csv', header=0,
            index_col=None, skipinitialspace=True)

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        main_setting.trainer.num_workers = util.determine_max_process(4)
        main_setting.trainer.non_blocking = False
        tr_workers_4_blocking = trainer.Trainer(main_setting)
        tr_workers_4_blocking.train()
        df_workers_4_blocking = pd.read_csv(
            main_setting.trainer.output_directory / 'log.csv', header=0,
            index_col=None, skipinitialspace=True)

        # Confirm multiprocess is faster
        self.assertLess(
            df_workers_4_blocking['elapsed_time'][-1],
            df_workers_0['elapsed_time'][-1])
        self.assertLess(
            df_workers_4['elapsed_time'][-1],
            df_workers_4_blocking['elapsed_time'][-1])

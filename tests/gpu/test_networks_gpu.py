from pathlib import Path
import shutil
import unittest

from chainer import testing
import numpy as np

import siml.setting as setting
import siml.trainer as trainer


class TestNetworksGPU(unittest.TestCase):

    @testing.attr.multi_gpu(2)
    def test_nri_gpu(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/nri.yml'))
        main_setting.trainer.gpu_id = 1
        main_setting.trainer.num_workers = 0  # Serial
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    @testing.attr.multi_gpu(2)
    def test_res_gcn_gpu(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/res_gcn.yml'))
        main_setting.trainer.gpu_id = 1
        main_setting.trainer.num_workers = 0  # Serial
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 5.)

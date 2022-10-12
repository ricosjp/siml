from pathlib import Path
import shutil
import unittest

import numpy as np

import siml.inferer as inferer
import siml.setting as setting
import siml.trainer as trainer


class TestNetworks(unittest.TestCase):

    def test_dggnn(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/advection/dggnn.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=main_setting.data.validation)

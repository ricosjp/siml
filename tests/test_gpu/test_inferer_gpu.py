from pathlib import Path
import shutil
import unittest

import numpy as np

import siml.setting as setting
import siml.inferer as inferer


class TestInfererGPU(unittest.TestCase):

    def test_infer_gpu(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/pretrained/settings.yml'))
        main_setting.inferer.converter_parameters_pkl = Path(
            'tests/data/linear/preprocessed/preprocessors.pkl')
        main_setting.inferer.output_directory_root = Path(
            'tests/data/linear/inferred')
        ir = inferer.Inferer(main_setting)
        ir.setting.inferer.gpu_id = 0
        if ir.setting.trainer.output_directory.exists():
            shutil.rmtree(ir.setting.trainer.output_directory)
        res = ir.infer(
            model=Path('tests/data/linear/pretrained'),
            data_directories=Path('tests/data/linear/preprocessed/validation'))
        np.testing.assert_almost_equal(
            res[0]['dict_y']['y'],
            np.load('tests/data/linear/interim/validation/0/y.npy'), decimal=2)
        np.testing.assert_array_less(res[0]['loss'], 1e-7)

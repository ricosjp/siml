from pathlib import Path
import shutil
import unittest

from chainer import testing
import numpy as np

import siml.optimize as optimize
import siml.setting as setting


class TestOptimizeGPU(unittest.TestCase):

    @testing.attr.multi_gpu(2)
    def test_perform_study_gpu(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/optuna.yml'))
        main_setting.trainer.gpu_id = 1
        main_setting.trainer.num_workers = 0  # Serial
        main_setting.optuna.n_trial = 2
        if main_setting.optuna.output_base_directory.exists():
            shutil.rmtree(main_setting.optuna.output_base_directory)

        study = optimize.Study(main_setting)
        study.perform_study()
        self.assertLess(
            study.study.best_trial.value,
            np.max([t.value for t in study.study.trials]))

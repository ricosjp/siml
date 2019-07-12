from pathlib import Path
import unittest

import chainer as ch
import numpy as np

import siml.setting as setting
import siml.trainer as trainer


class TestTrainer(unittest.TestCase):

    def test_train_cpu(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/linear/linear.yml')
        trainer.Trainer(main_setting)

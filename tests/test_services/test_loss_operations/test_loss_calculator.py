import pytest
from pathlib import Path
import torch
import numpy as np

import siml.setting as setting
import siml.trainer as trainer


@pytest.fixture
def initialize_trainer() -> trainer.Trainer:
    main_setting = setting.MainSetting.read_settings_yaml(
        Path('tests/data/deform/dict_input_user_loss.yml'))
    tr = trainer.Trainer(
        main_setting,
        user_loss_fundtion_dic={
            "user_mspe": lambda x, y:
            torch.mean(torch.square((x - y) / (torch.norm(y) + 0.0001)))
        })
    return tr


@pytest.mark.parametrize("x, y", [
    (torch.Tensor([3, 5]), torch.Tensor([4, 1])),
    (torch.Tensor([0, 1]), torch.Tensor([0, 0])),
    (torch.Tensor([-12.2, -43.2]), torch.Tensor([-20.2, -43.2])),
])
def test_check_user_loss_function(initialize_trainer, x, y):
    tr = initialize_trainer
    loss_calculator = tr._create_loss_function()

    loss_actual = loss_calculator.loss_core(x, y, "out_rank2_gauss1_2")
    loss_expect = torch.mean(torch.square((x - y) / (torch.norm(y) + 0.0001)))

    np.testing.assert_almost_equal(loss_actual, loss_expect)

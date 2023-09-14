from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

import siml.setting as setting
import siml.trainer as trainer
from siml.loss_operations import LossCalculatorBuilder
from siml.loss_operations.loss_assignment import LossAssignmentCreator
from siml.loss_operations.loss_calculator import CoreLossCalculator


@pytest.fixture
def initialize_calculator() -> trainer.Trainer:
    main_setting = setting.MainSetting.read_settings_yaml(
        Path('tests/data/deform/dict_input_user_loss.yml'))
    user_loss_function_dic = {
        "user_mspe": lambda x, y:
        torch.mean(torch.square((x - y) / (torch.norm(y) + 0.0001)))
    }
    loss = LossCalculatorBuilder.create(
        trainer_setting=main_setting.trainer,
        user_loss_function_dic=user_loss_function_dic
    )
    return loss


@pytest.mark.parametrize("x, y", [
    (torch.Tensor([3, 5]), torch.Tensor([4, 1])),
    (torch.Tensor([0, 1]), torch.Tensor([0, 0])),
    (torch.Tensor([-12.2, -43.2]), torch.Tensor([-20.2, -43.2])),
])
def test__check_user_loss_function(initialize_calculator, x, y):
    loss_calculator = initialize_calculator

    loss_actual = loss_calculator.loss_core(x, y, "out_rank2_gauss1_2")
    loss_expect = torch.mean(torch.square((x - y) / (torch.norm(y) + 0.0001)))

    np.testing.assert_almost_equal(loss_actual, loss_expect)


@pytest.mark.parametrize("x, y, weight", [
    (torch.Tensor([3, 5]), torch.Tensor([4, 1]), 2.0),
    (torch.Tensor([0, 1]), torch.Tensor([0, 0]), 3.0),
    (torch.Tensor([-12.2, -43.2]), torch.Tensor([-20.2, -43.2]), 3.0),
])
def test__core_loss_calculator_with_weight(x, y, weight):

    loss_assignment = LossAssignmentCreator.create("mse")
    core_loss_calculator = CoreLossCalculator(
        loss_assignment=loss_assignment,
        loss_weights={"var_x": weight}
    )

    actual: np.ndarray = core_loss_calculator(x, y, "var_x").numpy()
    expect: np.ndarray = weight * functional.mse_loss(x, y).numpy()

    np.testing.assert_almost_equal(
        actual.astype(np.float32), expect.astype(np.float32)
    )


@pytest.mark.parametrize('loss_weights', [
    {"var_x": 2.1, "var_y": 1.0},
    {"var_k": 3.2, "var_l": 4.5}
])
def test__get_loss_weight(loss_weights):
    loss_assignment = LossAssignmentCreator.create("mse")
    core_loss_calculator = CoreLossCalculator(
        loss_assignment=loss_assignment,
        loss_weights=loss_weights
    )

    for k, v in loss_weights.items():
        actual = core_loss_calculator._get_loss_weight(k)
        assert v == actual


@pytest.mark.parametrize('loss_weights, missing_name', [
    ({"var_x": 2.1, "var_y": 1.0}, "var_z"),
    ({"var_k": 3.2, "var_l": 4.5}, "var_x"),
])
def test__raise_error_when_get_loss_weight(loss_weights, missing_name):
    loss_assignment = LossAssignmentCreator.create("mse")
    core_loss_calculator = CoreLossCalculator(
        loss_assignment=loss_assignment,
        loss_weights=loss_weights
    )

    with pytest.raises(KeyError):
        core_loss_calculator._get_loss_weight(missing_name)


def test__get_loss_weight_when_none():
    loss_assignment = LossAssignmentCreator.create("mse")
    core_loss_calculator = CoreLossCalculator(
        loss_assignment=loss_assignment,
    )

    any_name = "var_x"
    actual = core_loss_calculator._get_loss_weight(any_name)
    assert actual == 1.0

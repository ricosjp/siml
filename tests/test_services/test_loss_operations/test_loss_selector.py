import pytest
import torch.nn.functional as functional
import numpy as np

from siml.loss_operations.loss_assignment import LossAssignmentCreator
from siml.loss_operations.loss_selector import LossFunctionSelector


@pytest.fixture
def preprare_default_inputs():
    # case 1
    loss_assignment = LossAssignmentCreator.create("mse")
    user_loss_function_dic = None
    return loss_assignment, user_loss_function_dic


@pytest.fixture
def prepare_user_valid_inputs():
    loss_assignment = LossAssignmentCreator.create({"value1": "USER_FUNC",
                                                    "value2": "mse"})
    user_loss_function_dic = {
        "USER_FUNC": lambda x, y: np.abs(x - y) / (x + 0.0001)
    }
    return loss_assignment, user_loss_function_dic


@pytest.fixture
def prepare_user_invalid_inputs():
    loss_assignment = LossAssignmentCreator.create({"value1": "USER_FUNC",
                                                    "value2": "mse"})
    user_loss_function_dic = {
        "USER_FUNC_1": lambda x, y: np.abs(x - y) / (x + 0.0001)
    }
    return loss_assignment, user_loss_function_dic


def test_can_select_function_for_default_inputs(preprare_default_inputs):
    loss_assignment, user_loss_function_dic = preprare_default_inputs
    selector = LossFunctionSelector(
        loss_assignment,
        user_loss_function_dic=user_loss_function_dic
    )

    assert selector.get_loss_function("value_any") == functional.mse_loss


def test_can_select_function_for_user_inputs(prepare_user_valid_inputs):
    loss_assignment, user_loss_function_dic = prepare_user_valid_inputs
    selector = LossFunctionSelector(
        loss_assignment,
        user_loss_function_dic=user_loss_function_dic
    )

    assert selector.get_loss_function("value1") == \
        user_loss_function_dic[loss_assignment["value1"]]
    assert selector.get_loss_function("value2") == functional.mse_loss


def test_can_detect_invalid_inputs(prepare_user_invalid_inputs):
    loss_assignment, user_loss_function_dic = prepare_user_invalid_inputs

    with pytest.raises(ValueError) as ex:
        _ = LossFunctionSelector(
            loss_assignment,
            user_loss_function_dic=user_loss_function_dic
        )

    assert str(ex.value) == "Unknown loss function name: USER_FUNC"

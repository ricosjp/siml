
import pytest

from siml.loss_operations.loss_assignment import LossAssignmentCreator


@pytest.mark.parametrize("loss_name", [
    'mse', 'mae', 'user_func_1'
])
def test__str_loss_assignment(loss_name):
    loss_assignment = LossAssignmentCreator().create(loss_name)
    assert loss_name == loss_assignment.loss_names[0]
    assert loss_name == loss_assignment["value_any"]


@pytest.mark.parametrize("loss_dict", [
    {"value_1": "mse", "value_2": "mae"},
    {"value_1": "mse", "value_2": "user_func_1"}
])
def test__dict_loss_assignment(loss_dict: dict):

    loss_assignment = LossAssignmentCreator().create(loss_dict)

    for k, v in loss_dict.items():
        assert v in loss_assignment.loss_names
        assert v == loss_assignment[k]

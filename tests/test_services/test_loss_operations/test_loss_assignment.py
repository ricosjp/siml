
import pytest
from siml.loss_operations.loss_assignment import LossAssignmentCreator


@pytest.fixture()
def generate_str_loss_settings():
    loss_settings = ['mse', 'mae', 'user_func_1']
    return loss_settings


@pytest.fixture()
def generate_dict_loss_settings():
    loss_settings = [
        {"value_1": "mse", "value_2": "mae"},
        {"value_1": "mse", "value_2": "user_func_1"}
    ]
    return loss_settings


@pytest.fixture()
def test_enable_to_generate_str_loss_assignment(generate_str_loss_settings):
    for loss_setting in generate_str_loss_settings:
        loss_assignment = LossAssignmentCreator().create(loss_setting)
        assert loss_setting == loss_assignment.loss_names[0]

        assert loss_setting == loss_assignment["value_any"]


@pytest.fixture()
def test_enable_to_generate_dict_loss_assignment(
        generate_dict_loss_settings):

    for loss_setting in generate_dict_loss_settings:
        loss_assignment = LossAssignmentCreator().create(loss_setting)

        for k, v in loss_setting.items():
            assert v in loss_assignment.loss_names
            assert v == loss_assignment[k]

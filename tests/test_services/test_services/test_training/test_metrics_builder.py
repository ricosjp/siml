from unittest import mock

import pytest
import numpy as np

from siml.services.training.metrics_builder import LossDetailsMetrics
from siml.loss_operations import ILossCalculator


@pytest.mark.parametrize("loss_details", [
    ({"var_x": np.array([[0.01]]), "var_y": np.array([[0.02]])})
])
def test__update_as_init_values(loss_details: dict[str, np.ndarray]):

    with mock.patch.multiple(
        ILossCalculator,  __abstractmethods__=set()
    ):
        mocked = ILossCalculator()
        mocked.calculate_loss_details = mock.MagicMock(
            return_value=loss_details
        )
        metrics = LossDetailsMetrics(mocked)
        metrics.reset()

        mock_input = [1, 1]
        metrics.update(mock_input)

        results = metrics.compute()
        assert len(results) == 2
        for k, v in results.items():
            assert loss_details[k].item() == v


@pytest.mark.parametrize("loss_details, expect", [
    (
        [
            {"var_x": np.array([[0.23]]), "var_y": np.array([[0.02]])},
            {"var_x": np.array([[0.13]]), "var_y": np.array([[0.02]])}
        ],
        {"var_x": 0.36 / 2.0, "var_y": 0.04 / 2.0}
    )
])
def test__update_as_multiple_values(
    loss_details: list[dict[str, np.ndarray]],
    expect: dict[str, np.ndarray]
):

    with mock.patch.multiple(
        ILossCalculator,  __abstractmethods__=set()
    ):
        mocked = ILossCalculator()
        mocked.calculate_loss_details = mock.MagicMock(
            side_effect=loss_details
        )
        metrics = LossDetailsMetrics(mocked)
        metrics.reset()

        mock_input = [1, 1]
        for _ in range(len(loss_details)):
            metrics.update(mock_input)

        results = metrics.compute()
        assert len(results) == len(expect)
        for k, v in results.items():
            assert v == expect[k]


def test__compute_assert_error():

    with mock.patch.multiple(
        ILossCalculator,  __abstractmethods__=set()
    ):
        mocked = ILossCalculator()
        metrics = LossDetailsMetrics(mocked)
        metrics.reset()

        with pytest.raises(ValueError):
            metrics.compute()

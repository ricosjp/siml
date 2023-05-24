from typing import Callable, Union
import pathlib

import pytest
import torch
import torch.nn.functional as functional
import numpy as np

from siml.setting import TrainerSetting
from siml.services.inference import metrics_builder
from siml.services.inference import PostPredictionRecord, PredictionRecord
from siml.siml_variables import siml_tensor_variables


def create_mocked_inputs(
    x_tensor: torch.Tensor,
    y_pred_tensor: torch.Tensor,
    y_tensor: Union[torch.Tensor, None],
    post_func: Callable[[np.ndarray], np.ndarray] = None
):
    if post_func is None:
        def post_func(x): return x
    record = PredictionRecord(
        x=siml_tensor_variables({"x": x_tensor}),
        y_pred=siml_tensor_variables({"y": y_pred_tensor}),
        y=siml_tensor_variables({"y": y_tensor})
        if y_tensor is not None else None,
        original_shapes=x_tensor.shape,
        inference_time=10.,
        data_directory=pathlib.Path("./sample/path")
    )

    # Equivalent to perform_inverse = False
    x_numpy = post_func(x_tensor.detach().cpu().numpy())
    y_pred_numpy = post_func(y_pred_tensor.detach().cpu().numpy())
    if y_tensor is None:
        dict_answer = None
    else:
        y_numpy = post_func(y_tensor.detach().cpu().numpy())
        dict_answer = {"y": y_numpy}
    post_record = PostPredictionRecord(
        dict_x={"x": x_numpy},
        dict_y={"y": y_pred_numpy},
        dict_answer=dict_answer,
        original_shapes=(1, 2),
        data_directory=pathlib.Path("./sample/path"),
        inference_time=10,
        inference_start_datetime="2023.01.01 12:00:00",
        fem_data=None
    )
    return (y_pred_tensor, y_tensor, {
        "result": record, "post_result": post_record
    })


def test__post_results_metrics():
    x_tensor = torch.tensor([1, 0])
    y_tensor = torch.tensor([1, 0])
    y_pred_tensor = torch.tensor([1, 0])
    output = create_mocked_inputs(
        x_tensor, y_pred_tensor, y_tensor
    )

    metrics = metrics_builder.PostResultsMetrics()
    metrics.update(output)
    results = metrics.compute()

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0] == output[2]["post_result"]


@pytest.mark.parametrize("x_tensor, y_tensor, y_pred_tensor", [
    ([[2, 3]], [[5., 3.]], [[4., 6.]])
])
def test__mae_loss_metrics(x_tensor, y_tensor, y_pred_tensor):
    x_tensor = torch.tensor(x_tensor)
    y_tensor = torch.tensor(y_tensor)
    y_pred_tensor = torch.tensor(y_pred_tensor)

    output = create_mocked_inputs(
        x_tensor, y_pred_tensor, y_tensor
    )

    def mae_loss(y_pred, y, original_shapes=None):
        return functional.l1_loss(y_pred, y)

    metrics = metrics_builder.LossMetrics(mae_loss)
    metrics.update(output)
    results = metrics.compute()

    expect = mae_loss(y_pred_tensor, y_tensor).detach().numpy().item()

    np.testing.assert_almost_equal(results[0], expect)


@pytest.mark.parametrize("x_tensor, y_tensor, y_pred_tensor", [
    ([[2, 3]], [[5., 3.]], [[4., 6.]])
])
def test__mae_raw_loss_metrics(x_tensor, y_tensor, y_pred_tensor):
    trainer_setting = TrainerSetting()
    x_tensor = torch.tensor(x_tensor)
    y_tensor = torch.tensor(y_tensor)
    y_pred_tensor = torch.tensor(y_pred_tensor)

    def post_func(x): return 10 * x

    output = create_mocked_inputs(
        x_tensor, y_pred_tensor, y_tensor, post_func
    )

    def mae_loss(y_pred, y, original_shapes=None):
        return functional.l1_loss(y_pred, y)

    metrics = metrics_builder.RawLossMetrics(
        trainer_setting,
        mae_loss
    )
    metrics.update(output)
    results = metrics.compute()

    expect = 10. * mae_loss(y_pred_tensor, y_tensor).detach().numpy().item()
    np.testing.assert_almost_equal(results[0], expect)


@pytest.mark.parametrize("x_tensor, y_tensor, y_pred_tensor", [
    ([[2, 3]], None, [[5., 3.]])
])
def test__mae_raw_loss_metrics_with_none(x_tensor, y_tensor, y_pred_tensor):
    trainer_setting = TrainerSetting()
    x_tensor = torch.tensor(x_tensor)
    y_pred_tensor = torch.tensor(y_pred_tensor)

    def post_func(x): return 10 * x

    output = create_mocked_inputs(
        x_tensor, y_pred_tensor, y_tensor, post_func
    )

    def mae_loss(y_pred, y, original_shapes=None):
        return functional.l1_loss(y_pred, y)

    metrics = metrics_builder.RawLossMetrics(
        trainer_setting,
        mae_loss
    )
    metrics.update(output)
    results = metrics.compute()

    assert results[0] is None

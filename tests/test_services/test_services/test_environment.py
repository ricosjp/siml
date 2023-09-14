import pytest
from unittest import mock

from siml.services import ModelEnvironmentSetting


@pytest.mark.parametrize("data_parallel, model_parallel", [
    (True, True),
    (True, False),
    (False, True)
])
def test__check_cpu_avaiable_not_allowed(data_parallel, model_parallel):
    with pytest.raises(ValueError):
        _ = ModelEnvironmentSetting(
            gpu_id=-1, seed=0, data_parallel=data_parallel,
            model_parallel=model_parallel, time_series=False
        )


@mock.patch("torch.cuda.is_available")
def test__check_not_cuda_allowed(mocked):
    mocked.return_value = False
    with pytest.raises(ValueError) as exc_info:
        _ = ModelEnvironmentSetting(
            gpu_id=0, seed=0, data_parallel=False,
            model_parallel=False, time_series=False
        )
    assert exc_info.value.args[0] == "No GPU found."


@mock.patch("torch.cuda.is_available")
def test__check_cuda_allowed(mocked):
    mocked.return_value = True
    _ = ModelEnvironmentSetting(
        gpu_id=0, seed=0, data_parallel=False,
        model_parallel=False, time_series=False
    )


@mock.patch("torch.cuda.is_available")
def test__check_gpu_supported_setting(mocked):
    mocked.return_value = True
    with pytest.raises(ValueError):
        _ = ModelEnvironmentSetting(
            gpu_id=0, seed=0, data_parallel=True,
            model_parallel=False, time_series=True
        )


@pytest.mark.parametrize(
    "gpu_id, data_parallel, model_parallel, device", [
        (-1, False, False, 'cpu'),
        (0, False, False, 'cuda:0'),
        (0, True, False, 'cuda:0'),
        (0, False, True, 'cuda:0'),
        (2, False, False, 'cuda:2')
    ])
def test__get_device(
    gpu_id, data_parallel, model_parallel, device
):
    with mock.patch("torch.cuda.is_available") as mocked:
        if gpu_id == -1:
            mocked.return_value = False
        else:
            mocked.return_value = True
        env_setting = ModelEnvironmentSetting(
            gpu_id=gpu_id, seed=0, data_parallel=data_parallel,
            model_parallel=model_parallel, time_series=False
        )

        assert env_setting.get_device() == device


@pytest.mark.parametrize(
    "gpu_id, data_parallel, model_parallel, output_device", [
        (-1, False, False, None),
        (0, False, False, 'cuda:0'),
        (0, True, False, 'cuda:0'),
        (2, False, False, 'cuda:2')
    ])
def test__get_output_device(
    gpu_id, data_parallel, model_parallel, output_device
):
    with mock.patch("torch.cuda.is_available") as mocked:
        if gpu_id == -1:
            mocked.return_value = False
        else:
            mocked.return_value = True

        env_setting = ModelEnvironmentSetting(
            gpu_id=gpu_id, seed=0, data_parallel=data_parallel,
            model_parallel=model_parallel, time_series=False
        )

        if output_device is None:
            assert env_setting.get_output_device() is None
        else:
            assert env_setting.get_output_device() == output_device


@pytest.mark.parametrize(
    "gpu_id, gpu_count, output_device", [
        (0, 1, 'cuda:0'),
        (0, 5, 'cuda:4'),
        (2, 2, 'cuda:1')
    ])
def test__get_output_device_model_parallel(
    gpu_id, gpu_count, output_device
):
    with mock.patch("torch.cuda.is_available") as mocked1, \
            mock.patch("torch.cuda.device_count") as mocked2:
        mocked1.return_value = True
        mocked2.return_value = gpu_count

        env_setting = ModelEnvironmentSetting(
            gpu_id=gpu_id, seed=0, data_parallel=False,
            model_parallel=True, time_series=False
        )

        assert env_setting.get_output_device() == output_device

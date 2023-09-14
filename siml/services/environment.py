import random
from typing import Tuple

import numpy as np
import pydantic.dataclasses as dcs
import torch


@dcs.dataclass(init=True, frozen=True)
class ModelEnvironmentSetting:
    """
    This class has information of model environments.
    In the future, this class is deprecated and integrated to MainSetting
    """
    gpu_id: int
    seed: int
    data_parallel: bool
    model_parallel: bool
    time_series: bool

    def __post_init_post_parse__(self) -> None:
        """
        Check settings
        """
        if self.gpu_id == -1:
            # CPU設定
            self._check_cpu_available_setting()
        else:
            # GPU設定
            self._check_gpu_available_setting()

    @property
    def gpu_count(self) -> int:
        return torch.cuda.device_count()

    def set_seed(self) -> None:
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return

    def get_device(self) -> str:
        device, _ = self._select_devices()
        return device

    def get_output_device(self) -> str:
        _, output_device = self._select_devices()
        return output_device

    def _select_devices(self) -> Tuple[str, str]:
        """Return devices (input_device, output_device)

        Returns
        -------
        Tuple[str, str]
            (input_device, output_device)
        """
        if self.gpu_id == -1:
            return 'cpu', None

        if self.data_parallel:
            return 'cuda:0', 'cuda:0'

        if self.model_parallel:
            return 'cuda:0', f"cuda:{self.gpu_count - 1}"

        return f"cuda:{self.gpu_id}", f"cuda:{self.gpu_id}"

    def _check_cpu_available_setting(self) -> None:
        if self.data_parallel:
            raise ValueError("Data parallel is not available in CPU")

        if self.model_parallel:
            raise ValueError("Model parallel is not available in CPU")

    def _check_gpu_available_setting(self) -> None:
        if not self._is_gpu_supporting():
            raise ValueError('No GPU found.')

        if self.data_parallel and self.time_series:
            raise ValueError(
                'So far both data_parallel and time_series cannot be '
                'True'
            )

    def _is_gpu_supporting(self):
        return torch.cuda.is_available()

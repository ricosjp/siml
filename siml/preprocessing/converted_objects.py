from __future__ import annotations
from typing import Optional, Union

import femio
import numpy as np

from enum import Enum


class ConvertedStatus(Enum):
    not_finished = 0
    successed = 1
    failed = 2
    skipped = 3


class SimlConvertedItemContainer:
    def __init__(self, values: dict[str, SimlConvertedItem]) -> None:
        self._values = values

    def __getitem__(self, name: str) -> SimlConvertedItem:
        return self._values[name]

    def select_successed(
        self
    ) -> dict[str, tuple[
        Union[dict[str, np.ndarray], None],
        Union[femio.FEMData, None]
    ]]:
        vals = {
            k: v.get_values()
            for k, v in self._values.items() if v.is_successed
        }
        return vals

    def select_failed_items(self) -> dict[str, SimlConvertedItem]:
        vals = {
            k: v
            for k, v in self._values.items() if not v.is_successed
        }
        return vals


class SimlConvertedItem:
    def __init__(self) -> None:
        self._status = ConvertedStatus.not_finished
        self._dict_data: Union[dict[str, np.ndarray], None] = None
        self._fem_data: Union[femio.FEMData, None] = None

        self._failed_message: str = ""

    def skipped(self) -> None:
        self._status = ConvertedStatus.skipped

    def failed(self, message: str) -> None:
        self._status = ConvertedStatus.failed
        self._failed_message = message

    def successed(self) -> None:
        self._status = ConvertedStatus.successed
        self._successed = True

    def register(
        self,
        *,
        dict_data: Optional[dict[str, np.ndarray]],
        fem_data: Optional[femio.FEMData]
    ) -> None:
        if self._dict_data is not None:
            raise ValueError(
                'dict_data has already been registered. '
                'Not allowed to overwrite.'
            )

        if self._fem_data is not None:
            raise ValueError(
                'fem_data has already been registered. '
                'Not allowed to overwrite.'
            )

        self._dict_data = dict_data
        self._fem_data = fem_data

    @property
    def is_successed(self):
        return self._successed == ConvertedStatus.successed

    @property
    def is_failed(self):
        return self._status == ConvertedStatus.failed

    @property
    def is_skipped(self):
        return self._status == ConvertedStatus.skipped

    def get_values(
        self
    ) -> tuple[
        Union[dict[str, np.ndarray], None],
        Union[femio.FEMData, None]
    ]:
        return (self._dict_data, self._fem_data)

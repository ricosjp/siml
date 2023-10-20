from __future__ import annotations

from enum import Enum
from typing import Optional, Union

import femio
import numpy as np
from _collections_abc import dict_keys


class SimlConvertedStatus(Enum):
    not_finished = 0
    successed = 1
    failed = 2
    skipped = 3


class SimlConvertedItemContainer:
    def __init__(self, values: dict[str, SimlConvertedItem]) -> None:
        self._values = values

    def __len__(self):
        return len(self._values)

    def keys(self) -> dict_keys[str]:
        return self._values.keys()

    def __getitem__(self, name: str) -> SimlConvertedItem:
        return self._values[name]

    @property
    def is_all_successed(self) -> bool:
        items = self.select_non_successed_items()
        return len(items) == 0

    def query_num_status_items(self, *status: str) -> int:
        """query the number of data which has the status.

        Returns
        -------
        int
            number of data to be selected

        Raises
        ------
        ValueError
            If status is not defined, raise this error.
        """
        for s in status:
            if s not in SimlConvertedStatus.__members__:
                raise ValueError(f"status name: {s} is not defined.")

        n_vals = len([
            v for v in self._values.values()
            if v.get_status() in status
        ])
        return n_vals

    def merge(
        self, other: SimlConvertedItemContainer
    ) -> SimlConvertedItemContainer:
        """return new object merging self data and others.
        if same key exists in both objects, key in other is prioritised.

        Parameters
        ----------
        other : SimlConvertedItemContainer
            container to merge

        Returns
        -------
        SimlConvertedItemContainer
            new container object which has merged data
        """
        new_values = self._values | other._values
        return SimlConvertedItemContainer(new_values)

    def select_successed_items(
        self
    ) -> dict[str, SimlConvertedItem]:
        """Select items of which status is successed.

        Returns
        -------
        dict[str, SimlConvertedItem]
            successed items
        """
        vals = {
            k: v for k, v in self._values.items() if v.is_successed
        }
        return vals

    def select_non_successed_items(self) -> dict[str, SimlConvertedItem]:
        """Select items of which status is not successed, such as failed,
         skipped, unfinished.

        Returns
        -------
        dict[str, SimlConvertedItem]
            non successed items
        """
        vals = {
            k: v
            for k, v in self._values.items() if not v.is_successed
        }
        return vals


class SimlConvertedItem:
    def __init__(self) -> None:
        self._status = SimlConvertedStatus.not_finished
        self._dict_data: Union[dict[str, np.ndarray], None] = None
        self._fem_data: Union[femio.FEMData, None] = None

        self._failed_message: str = ""

    def skipped(self, message: Optional[str] = None) -> None:
        """Set status as skipped
        """
        self._status = SimlConvertedStatus.skipped
        if message is None:
            return
        self._failed_message = "".join(message.splitlines())

    def failed(self, message: Optional[str] = None) -> None:
        """Set status as failed

        Parameters
        ----------
        message : Optional[str]
            If fed, register failed message
        """
        self._status = SimlConvertedStatus.failed
        if message is None:
            return
        self._failed_message = "".join(message.splitlines())

    def successed(self) -> None:
        """Set status as successed
        """
        self._status = SimlConvertedStatus.successed

    def register(
        self,
        *,
        dict_data: Optional[dict[str, np.ndarray]],
        fem_data: Optional[femio.FEMData]
    ) -> None:
        """Register result items

        Parameters
        ----------
        dict_data : Optional[dict[str, np.ndarray]]
            dict data of features
        fem_data : Optional[femio.FEMData]
            femio data

        Raises
        ------
        ValueError
            If dict_data has been already registered. raise this error.
        ValueError
            If fem_data has been already registered. raise this error.
        """
        if not self.is_successed:
            raise ValueError(
                'Status is not successed. '
                'Please call "successed" method beforehand.'
            )
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
        return self._status == SimlConvertedStatus.successed

    @property
    def is_failed(self):
        return self._status == SimlConvertedStatus.failed

    @property
    def is_skipped(self):
        return self._status == SimlConvertedStatus.skipped

    def get_status(self) -> str:
        return self._status.name

    def get_failed_message(self) -> str:
        return self._failed_message

    def get_values(
        self
    ) -> tuple[
        Union[dict[str, np.ndarray], None],
        Union[femio.FEMData, None]
    ]:
        """Get items which this object manages

        Returns
        -------
        tuple[ Union[dict[str, np.ndarray], None], Union[femio.FEMData, None] ]
            Return dict_data and fem_data if necessary
        """
        return (self._dict_data, self._fem_data)

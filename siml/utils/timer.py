import time
from typing import Final


class SimlStopWatch():
    def __init__(self, offset: float = 0.0) -> None:
        self._offset: Final[float] = offset
        self._start: float = None
        self._elapsed_time: float = offset

    def start(self) -> None:
        if self._start is not None:
            raise ValueError(
                "siml timer has already started."
            )
        self._start = time.time()

    def stop(self) -> float:
        if self._start is None:
            raise ValueError("timer has not started yet.")

        val = time.time() - self._start
        self._elapsed_time += val
        self._start = None
        return self._elapsed_time

    def reset(self) -> None:
        self._start = None
        self._elapsed_time = self._offset

    def watch(self) -> float:
        """Return elapsed time since timer started..

        Returns
        -------
        float
            Elapsed time from start time

        Raises
        ------
        ValueError
            If timer has not started, raise this error
        """
        if self._start is None:
            raise ValueError("timer has not started yet.")

        val = time.time() - self._start
        return val + self._elapsed_time

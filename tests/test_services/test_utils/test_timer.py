from unittest import mock

import pytest

from siml.utils.timer import SimlStopWatch


@pytest.mark.parametrize("offset, times, expected", [
    (0, [0, 100], 100),
    (10, [0, 100], 110)
])
def test__stop_function_elapsed_time(offset, times, expected):
    timer = SimlStopWatch(offset)
    with mock.patch("time.time", side_effect=times):
        timer.start()
        assert timer.stop() == expected
        assert timer._start is None


@pytest.mark.parametrize("offset, times, expected", [
    (0, [0, 100], 100),
    (10, [0, 100], 110)
])
def test__watch_function_elapsed_time(offset, times, expected):
    timer = SimlStopWatch(offset)
    with mock.patch("time.time", side_effect=times):
        timer.start()
        assert timer.watch() == expected
        assert timer._start is not None


def test__cannot_start():
    timer = SimlStopWatch()
    with pytest.raises(ValueError):
        timer.start()
        timer.start()


def test__cannot_stop():
    timer = SimlStopWatch()
    with pytest.raises(ValueError):
        timer.stop()


def test__cannot_watch():
    timer = SimlStopWatch()
    with pytest.raises(ValueError):
        timer.watch()

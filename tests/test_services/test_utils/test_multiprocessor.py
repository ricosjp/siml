import pytest
import os
import sys
import time
import numpy as np

from siml.utils.multiprocessor import (
    _process_chunk, _get_chunks, SimlMultiprocessor
)
from siml.utils.errors import SimlMultiprocessError


@pytest.mark.parametrize("fn, chunk, expects", [
    (
        lambda x: x * 10,
        [(1,), (2,), (3,)],
        [10, 20, 30]
    ),
    (
        lambda x, y: x * 10 + y,
        [(1, 1), (2, 2), (3, 5)],
        [11, 22, 35]
    )
])
def test__process_chunk(fn, chunk, expects):
    actual = _process_chunk(fn, chunk)
    assert actual == expects


@pytest.mark.parametrize("iterables, chunksize, expects", [
    (
        [i for i in range(10)],
        2,
        [((i,), (i + 1,)) for i in range(10) if i % 2 == 0]
    ),
    (
        [(i, i + 1) for i in range(10)],
        2,
        [(((i, i + 1),), ((i + 1, i + 2),)) for i in range(10) if i % 2 == 0]
    )
])
def test__get_chunks(iterables, chunksize, expects):

    for i, chunk in enumerate(_get_chunks(iterables, chunksize=chunksize)):
        assert chunk == expects[i]


@pytest.mark.parametrize("iterables, chunksize, expects", [
    (
        [[i for i in range(10)], [i + 1 for i in range(10)]],
        2,
        [((i, i + 1), (i + 1, i + 2)) for i in range(10) if i % 2 == 0]
    )
])
def test__get_chunks_multiples(iterables, chunksize, expects):

    for i, chunk in enumerate(_get_chunks(*iterables, chunksize=chunksize)):
        assert chunk == expects[i]


def freaky_job(num: int):
    if num == 1:
        sys.exit(0)
    else:
        return num


@pytest.mark.parametrize("inputs, expects", [
    (
        [3, 5, 6],
        [3, 5, 6]
    )
])
def test__can_execute_functions(inputs, expects):
    results = SimlMultiprocessor.run(
        inputs,
        max_process=2,
        target_fn=freaky_job
    )

    assert results == expects


@pytest.mark.parametrize("inputs", [
    ([3, 1, 6]),
    ([1, 1, 1])
])
def test__can_detect_child_process_error(inputs):
    with pytest.raises(SimlMultiprocessError):
        _ = SimlMultiprocessor.run(
            inputs,
            max_process=2,
            target_fn=freaky_job
        )


def sample_sleep_job(num: int):
    time.sleep(num)
    return num


@pytest.mark.parametrize("max_process, inputs, expects", [
    (2, [2, 2, 2, 2], 4),
])
def test__can_use_multi_core(max_process, inputs, expects):
    cpu_count = os.cpu_count()
    assert cpu_count >= max_process

    start = time.time()
    _ = SimlMultiprocessor.run(
        inputs,
        max_process=max_process,
        target_fn=sample_sleep_job,
    )
    elapsed_time = time.time() - start

    np.testing.assert_approx_equal(elapsed_time, expects, significant=1)


@pytest.mark.parametrize("max_process, inputs, chunksize, expects", [
    (2, [2, 2, 2, 2], 3, 6),
    (2, [1, 1, 1, 1], 2, 2)
])
def test__can_consider_chunksize(max_process, inputs, chunksize, expects):
    cpu_count = os.cpu_count()
    assert cpu_count >= max_process

    start = time.time()
    _ = SimlMultiprocessor.run(
        inputs,
        max_process=max_process,
        target_fn=sample_sleep_job,
        chunksize=chunksize,
    )
    elapsed_time = time.time() - start

    np.testing.assert_approx_equal(elapsed_time, expects, significant=1)


def sample_add(num1: int, num2: int):
    return num1 + num2


@pytest.mark.parametrize("max_process, inputs, chunksize, expects", [
    (2, [[2, 3, 4, 5], [2, 3, 4, 5]], 3, [4, 6, 8, 10]),
    (2, [[1, 1, 1, 1], [2, 3, 5, 6]], 2, [3, 4, 6, 7])
])
def test__can_flatten_return_objects(max_process, inputs, chunksize, expects):
    cpu_count = os.cpu_count()
    assert cpu_count >= max_process

    results = SimlMultiprocessor.run(
        *inputs,
        max_process=max_process,
        target_fn=sample_add,
        chunksize=chunksize,
    )

    assert results == expects

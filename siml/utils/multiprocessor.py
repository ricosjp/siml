import concurrent.futures as cf
import itertools
from functools import partial
from typing import Any, Callable, Iterable, TypeVar

from siml.utils.errors import SimlMultiprocessError


def _process_chunk(fn: Callable, chunk: list[Iterable[Any]]):
    """ Processes a chunk of an iterable passed to map.

    Runs the function passed to map() on a chunk of the
    iterable passed to map.

    This function is run in a separate process.

    """
    return [fn(*args) for args in chunk]


def _get_chunks(*iterables, chunksize):
    """ Iterates over zip()ed iterables in chunks. """
    it = zip(*iterables)
    while True:
        chunk = tuple(itertools.islice(it, chunksize))
        if not chunk:
            return
        yield chunk


def _santize_futures(futures: list[cf.Future]) -> None:
    for future in futures:
        ex = future.exception()
        if ex is None:
            continue

        raise SimlMultiprocessError(
            "Some jobs are failed under multiprocess conditions. "
            f"Exception: {ex}. "
            "If exeception content is only a integer number like 1, "
            "usage memory maybe be insufficient to run."
        )


T = TypeVar("T")


class SimlMultiprocessor():
    @staticmethod
    def run(
        max_process: int,
        target_fn: Callable[[Any], T],
        inputs: list[Any],
        *,
        chunksize: int
    ) -> list[T]:
        futures: list[cf.Future] = []
        with cf.ProcessPoolExecutor(max_process) as executor:
            for chunk in _get_chunks(inputs, chunksize=chunksize):
                future = executor.submit(
                    partial(_process_chunk, target_fn),
                    chunk
                )

                futures.append(future)

            cf.wait(futures)

        _santize_futures(futures)

        return itertools.chain.from_iterable([f.result() for f in futures])

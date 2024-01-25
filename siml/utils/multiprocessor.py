import concurrent.futures as cf
import itertools
from functools import partial
from typing import Any, Callable, Iterable, TypeVar

from siml.utils.errors import SimlMultiprocessError


T = TypeVar("T")


def _process_chunk(
    fn: Callable[[Any], T],
    chunk: list[Iterable[Any]]
) -> list[T]:
    """ Processes a chunk of an iterable passed to map.

    Runs the function passed to map() on a chunk of the
    iterable passed to map.

    This function is run in a separate process.

    """
    return [fn(*args) for args in chunk]


def _get_chunks(
    *iterables: Iterable[Any],
    chunksize: int
) -> Iterable[list[Any]]:
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
            "If content of exception shown above is only a integer number "
            "such as '1', it means that child process is killed by host system"
            " like OOM killer."
        )


class SimlMultiprocessor():
    def __init__(self, max_process: int):
        self.max_process = max_process

    def run(
        self,
        *inputs: list[Any],
        target_fn: Callable[[Any], T],
        chunksize: int = 1
    ) -> list[T]:
        """Wrapper function for concurrent.futures
         to run safely with multiple processes.

        Parameters
        ----------
        max_process : int
            the number of processes to use
        target_fn : Callable[[Any], T]
            function to execute
        chunksize : int, optional
            chunck size, by default 1

        Returns
        -------
        list[T]
            Iterable of objects returned from target_fn

        Raises
        -------
        SimlMultiprocessError:
            If some processes are killed by host system such as OOM killer,
             this error raises.
        """
        futures: list[cf.Future] = []
        with cf.ProcessPoolExecutor(self.max_process) as executor:
            for chunk in _get_chunks(*inputs, chunksize=chunksize):
                future = executor.submit(
                    partial(_process_chunk, target_fn),
                    chunk
                )

                futures.append(future)

            cf.wait(futures)

        _santize_futures(futures)

        # flatten
        return sum([f.result() for f in futures], start=[])

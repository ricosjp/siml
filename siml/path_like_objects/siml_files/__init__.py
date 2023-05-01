# flake8: noqa
from .interface import (
    ISimlCheckpointFile,
    ISimlPklFile,
    ISimlNumpyFile
)


from .numpy_file import SimlNpyEncFile, SimlNpyFile, SimlNpzEncFile, SimlNpzFile
from .pickle_file import SimlPklFile, SimlPklEncFile


NUMPY_FILES: list[ISimlNumpyFile] = [
    SimlNpyEncFile,
    SimlNpyFile,
    SimlNpzEncFile,
    SimlNpzFile
]

PICKLE_FILES: list[ISimlPklFile] = [
    SimlPklFile,
    SimlPklEncFile
]

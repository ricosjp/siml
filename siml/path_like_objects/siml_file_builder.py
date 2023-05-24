import pathlib

from .siml_files import (
    ISimlNumpyFile,
    ISimlPickleFile,
    ISimlCheckpointFile,
    ISimlYamlFile,
    SimlNumpyFile,
    SimlPickleFile,
    SimlCheckpointFile,
    SimlYamlFile
)


class SimlFileBuilder:
    @staticmethod
    def numpy_file(file_path: pathlib.Path) -> ISimlNumpyFile:
        return SimlNumpyFile(file_path)

    @staticmethod
    def pickle_file(file_path: pathlib.Path) -> ISimlPickleFile:
        return SimlPickleFile(file_path)

    @staticmethod
    def checkpoint_file(file_path: pathlib.Path) -> ISimlCheckpointFile:
        return SimlCheckpointFile(file_path)

    @staticmethod
    def yaml_file(file_path: pathlib.Path) -> ISimlYamlFile:
        return SimlYamlFile(file_path)

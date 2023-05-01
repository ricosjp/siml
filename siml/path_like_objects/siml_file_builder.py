import pathlib

from .siml_files import NUMPY_FILES, PICKLE_FILES, ISimlNumpyFile, ISimlPklFile


class SimlFileBulider:
    @staticmethod
    def numpy_file(file_path: pathlib.Path) -> ISimlNumpyFile:
        for file_cls in NUMPY_FILES:
            if str(file_path).endswith(file_cls.get_file_extension()):
                return file_cls(file_path)

        raise ValueError(f"File type not understood: {file_path}")

    @staticmethod
    def pickle_file(file_path: pathlib.Path) -> ISimlPklFile:
        for file_cls in PICKLE_FILES:
            if str(file_path).endswith(file_cls.get_file_extension()):
                return file_cls(file_path)

        raise ValueError(f"File type not understood: {file_path}")

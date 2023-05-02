import abc
import pathlib
import pickle
from enum import Enum
from typing import Any

import numpy as np
import scipy.sparse as sp

from siml import util


class SimlFileExtType(Enum):
    NPY = ".npy"
    NPYENC = ".npy.enc"
    NPZ = ".npz"
    NPZENC = ".npz.enc"
    PKL = ".pkl"
    PKLENC = ".pkl.enc"


class ISimlFile(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, path: pathlib.Path) -> None:
        raise NotImplementedError()

    @abc.abstractclassmethod
    def get_file_extension(cls) -> str:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def file_path(self) -> pathlib.Path:
        raise NotImplementedError()

    @abc.abstractmethod
    def load(
        self,
        check_nan: bool = False,
        decrypt_key: bytes = None
    ) -> Any:
        raise NotImplementedError()


class SimlNpyFile(ISimlFile):
    def __init__(self, path: pathlib.Path) -> None:
        assert str(path).endswith(SimlFileExtType.NPY.value)
        self._path = path

    def __str__(self) -> str:
        return f"SimlNpyFile: {self._path}"

    @classmethod
    def get_file_extension(cls) -> str:
        return SimlFileExtType.NPY.value

    @property
    def file_path(self) -> pathlib.Path:
        return self._path

    def load(
        self,
        check_nan: bool = False,
        decrypt_key: bytes = None
    ) -> np.ndarray:
        loaded_data = np.load(self._path)

        if check_nan and np.any(np.isnan(loaded_data)):
            raise ValueError(
                f"NaN found in {self._path}")

        return loaded_data


class SimlNpyEncFile(ISimlFile):
    def __init__(self, path: pathlib.Path) -> None:
        assert str(path).endswith(SimlFileExtType.NPYENC.value)
        self._path = path

    def __str__(self) -> str:
        return f"SimlNpyEncFile: {self._path}"

    @classmethod
    def get_file_extension(cls) -> str:
        return SimlFileExtType.NPYENC.value

    @property
    def file_path(self) -> pathlib.Path:
        return self._path

    def load(
        self,
        check_nan: bool = False,
        decrypt_key: bytes = None
    ) -> np.ndarray:
        loaded_data = np.load(
            util.decrypt_file(decrypt_key, self._path)
        )

        if check_nan and np.any(np.isnan(loaded_data)):
            raise ValueError(
                f"NaN found in {self._path}"
            )

        return loaded_data


class SimlNpzFile(ISimlFile):
    def __init__(self, path: pathlib.Path) -> None:
        assert str(path).endswith(SimlFileExtType.NPZ.value)
        self._path = path

    def __str__(self) -> str:
        return f"SimlNpzFile: {self._path}"

    @classmethod
    def get_file_extension(cls) -> str:
        return SimlFileExtType.NPZ.value

    @property
    def file_path(self) -> pathlib.Path:
        return self._path

    def load(
        self,
        check_nan: bool = False,
        decrypt_key: bytes = None
    ) -> np.ndarray:
        loaded_data = sp.load_npz(self._path)

        if check_nan and np.any(np.isnan(loaded_data)):
            raise ValueError(
                f"NaN found in {self._path}")

        return loaded_data


class SimlNpzEncFile(ISimlFile):
    def __init__(self, path: pathlib.Path) -> None:
        assert str(path).endswith(SimlFileExtType.NPZENC.value)
        self._path = path

    def __str__(self) -> str:
        return f"SimlNpzEncFile: {self._path}"

    @classmethod
    def get_file_extension(cls) -> str:
        return SimlFileExtType.NPZENC.value

    @property
    def file_path(self) -> pathlib.Path:
        return self._path

    def load(
        self,
        check_nan: bool = False,
        decrypt_key: bytes = None
    ) -> np.ndarray:
        loaded_data = sp.load_npz(
            util.decrypt_file(decrypt_key, self._path)
        )

        if check_nan and np.any(np.isnan(loaded_data)):
            raise ValueError(
                f"NaN found in {self._path}")

        return loaded_data


class SimlPklFile(ISimlFile):
    def __init__(self, path: pathlib.Path) -> None:
        assert str(path).endswith(SimlFileExtType.PKL.value)
        self._path = path

    def __str__(self) -> str:
        return f"SimlPklFile: {self._path}"

    @classmethod
    def get_file_extension(self) -> str:
        return SimlFileExtType.PKL.value

    @property
    def file_path(self) -> pathlib.Path:
        return self._path

    def load(
        self,
        check_nan: bool = False,
        decrypt_key: bytes = None
    ) -> np.ndarray:
        with open(self._path, 'rb') as f:
            parameters = pickle.load(f)
        return parameters


class SimlPklEncFile(ISimlFile):
    def __init__(self, path: pathlib.Path) -> None:
        assert str(path).endswith(SimlFileExtType.PKLENC.value)
        self._path = path

    def __str__(self) -> str:
        return f"SimlPklEncFile: {self._path}"

    @classmethod
    def get_file_extension(self) -> str:
        return SimlFileExtType.PKLENC.value

    @property
    def file_path(self) -> pathlib.Path:
        return self._path

    def load(
        self,
        check_nan: bool = False,
        decrypt_key: bytes = None
    ) -> np.ndarray:

        if decrypt_key is None:
            raise ValueError(
                "Key is None. Cannot decrypt encrypted file."
            )

        parameters = pickle.load(
            util.decrypt_file(decrypt_key, self._path)
        )
        return parameters


class SimlFileBulider:
    FILE_OBJECTS: list[ISimlFile] = [
        SimlNpyFile,
        SimlNpyEncFile,
        SimlNpzFile,
        SimlNpzEncFile,
        SimlPklFile,
        SimlPklEncFile
    ]

    @staticmethod
    def create(file_path: pathlib.Path) -> ISimlFile:
        for file_cls in SimlFileBulider.FILE_OBJECTS:
            if str(file_path).endswith(file_cls.get_file_extension()):
                return file_cls(file_path)

        raise ValueError(f"File type not understood: {file_path}")

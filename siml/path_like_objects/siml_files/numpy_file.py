import pathlib

import scipy.sparse as sp
import numpy as np

from siml.base.siml_enums import SimlFileExtType
from siml import util

from .interface import ISimlNumpyFile


class SimlNpyFile(ISimlNumpyFile):
    def __init__(self, path: pathlib.Path) -> None:
        assert str(path).endswith(SimlFileExtType.NPY.value)
        self._path = path

    def __str__(self) -> str:
        return f"{SimlNpyFile.__name__}: {self._path}"

    def is_encrypted(self) -> bool:
        return False

    @classmethod
    def get_file_extension(cls) -> str:
        return SimlFileExtType.NPY.value

    @property
    def file_path(self) -> pathlib.Path:
        return self._path

    def load(
        self,
        *,
        check_nan: bool = False,
        decrypt_key: bytes = None
    ) -> np.ndarray:
        loaded_data = np.load(self._path)

        if check_nan and np.any(np.isnan(loaded_data)):
            raise ValueError(
                f"NaN found in {self._path}")

        return loaded_data


class SimlNpyEncFile(ISimlNumpyFile):
    def __init__(self, path: pathlib.Path) -> None:
        assert str(path).endswith(SimlFileExtType.NPYENC.value)
        self._path = path

    def __str__(self) -> str:
        return f"{SimlNpyEncFile.__name__}: {self._path}"

    def is_encrypted(self) -> bool:
        return True

    @classmethod
    def get_file_extension(cls) -> str:
        return SimlFileExtType.NPYENC.value

    @property
    def file_path(self) -> pathlib.Path:
        return self._path

    def load(
        self,
        *,
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


class SimlNpzFile(ISimlNumpyFile):
    def __init__(self, path: pathlib.Path) -> None:
        assert str(path).endswith(SimlFileExtType.NPZ.value)
        self._path = path

    def is_encrypted(self) -> bool:
        return False

    def __str__(self) -> str:
        return f"{SimlNpzFile.__name__}: {self._path}"

    @classmethod
    def get_file_extension(cls) -> str:
        return SimlFileExtType.NPZ.value

    @property
    def file_path(self) -> pathlib.Path:
        return self._path

    def load(
        self,
        *,
        check_nan: bool = False,
        decrypt_key: bytes = None
    ) -> np.ndarray:
        loaded_data = sp.load_npz(self._path)

        if check_nan and np.any(np.isnan(loaded_data)):
            raise ValueError(
                f"NaN found in {self._path}")

        return loaded_data


class SimlNpzEncFile(ISimlNumpyFile):
    def __init__(self, path: pathlib.Path) -> None:
        assert str(path).endswith(SimlFileExtType.NPZENC.value)
        self._path = path

    def __str__(self) -> str:
        return f"{SimlNpzEncFile.__name__}: {self._path}"

    def is_encrypted(self) -> bool:
        return True

    @classmethod
    def get_file_extension(cls) -> str:
        return SimlFileExtType.NPZENC.value

    @property
    def file_path(self) -> pathlib.Path:
        return self._path

    def load(
        self,
        *,
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

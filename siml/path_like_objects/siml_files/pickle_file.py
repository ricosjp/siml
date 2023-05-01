import pathlib
import pickle

from siml.base.siml_enums import SimlFileExtType
from siml import util

from .interface import ISimlPklFile


class SimlPklFile(ISimlPklFile):
    def __init__(self, path: pathlib.Path) -> None:
        assert str(path).endswith(SimlFileExtType.PKL.value)
        self._path = path

    def __str__(self) -> str:
        return f"{SimlPklFile.__name__}: {self._path}"

    @classmethod
    def get_file_extension(self) -> str:
        return SimlFileExtType.PKL.value

    @property
    def file_path(self) -> pathlib.Path:
        return self._path

    def load(
        self,
        *,
        decrypt_key: bytes = None
    ) -> dict:
        with open(self._path, 'rb') as f:
            parameters = pickle.load(f)
        return parameters


class SimlPklEncFile(ISimlPklFile):
    def __init__(self, path: pathlib.Path) -> None:
        assert str(path).endswith(SimlFileExtType.PKLENC.value)
        self._path = path

    def __str__(self) -> str:
        return f"{SimlPklEncFile.__name__}: {self._path}"

    @classmethod
    def get_file_extension(self) -> str:
        return SimlFileExtType.PKLENC.value

    @property
    def file_path(self) -> pathlib.Path:
        return self._path

    def load(
        self,
        *,
        decrypt_key: bytes = None
    ) -> dict:

        if decrypt_key is None:
            raise ValueError(
                "Key is None. Cannot decrypt encrypted file."
            )

        parameters = pickle.load(
            util.decrypt_file(decrypt_key, self._path)
        )
        return parameters

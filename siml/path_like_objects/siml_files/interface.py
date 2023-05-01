import abc
import pathlib

from siml.base.siml_typing import ArrayDataType


class ISimlBaseFile(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, path: pathlib.Path) -> None:
        raise NotImplementedError()

    @abc.abstractclassmethod
    def get_file_extension(cls) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def is_encrypted(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    @property.getter
    @abc.abstractmethod
    def file_path(self) -> pathlib.Path:
        raise NotImplementedError()


class ISimlNumpyFile(ISimlBaseFile, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load(
        self,
        *,
        check_nan: bool = False,
        decrypt_key: bytes = None
    ) -> ArrayDataType:
        raise NotImplementedError()


class ISimlPklFile(ISimlBaseFile, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load(
        self,
        *,
        decrypt_key: bytes = None
    ) -> dict:
        raise NotImplementedError()


class ISimlCheckpointFile(ISimlBaseFile, metaclass=abc.ABCMeta):
    @property.getter
    def epoch(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def load(
        self,
        *,
        decrypt_key: bytes = None
    ) -> dict:
        raise NotImplementedError()

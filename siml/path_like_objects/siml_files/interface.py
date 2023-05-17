import abc
import pathlib

from siml.base.siml_typing import ArrayDataType


class ISimlBaseFile(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, path: pathlib.Path) -> None:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def is_encrypted(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def file_path(self) -> pathlib.Path:
        raise NotImplementedError()


class ISimlNumpyFile(ISimlBaseFile, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def file_extension(self) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def load(
        self,
        *,
        check_nan: bool = False,
        decrypt_key: bytes = None
    ) -> ArrayDataType:
        raise NotImplementedError()


class ISimlPickleFile(ISimlBaseFile, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load(
        self,
        *,
        decrypt_key: bytes = None
    ) -> dict:
        raise NotImplementedError()

    @abc.abstractmethod
    def save(
        self,
        dump_data: object,
        overwrite: bool = False,
        encrypt_key: bytes = None
    ) -> None:
        raise NotImplementedError()


class ISimlYamlFile(ISimlBaseFile, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load(
        self,
        *,
        decrypt_key: bytes = None
    ) -> dict:
        raise NotImplementedError()

    @abc.abstractmethod
    def save(
        self,
        dump_data: object,
        overwrite: bool = False,
        encrypt_key: bytes = None
    ) -> None:
        raise NotImplementedError()


class ISimlCheckpointFile(ISimlBaseFile, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def epoch(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def load(
        self,
        device: str,
        *,
        decrypt_key: bytes = None
    ) -> dict:
        raise NotImplementedError()

    @abc.abstractmethod
    def save(
        self,
        dump_data: object,
        overwrite: bool = False,
        encrypt_key: bytes = None
    ) -> None:
        raise NotImplementedError()

import pathlib
from typing import Any
import io
import re

import torch

from siml import util
from siml.base.siml_enums import SimlFileExtType

from .interface import ISimlCheckpointFile


class SimlCheckpointFile(ISimlCheckpointFile):
    def __init__(self, path: pathlib.Path):
        self._ext_type = self._check_extension_type(path)
        self._path = path

    def _check_extension_type(self, path: pathlib.Path) -> SimlFileExtType:
        extensions = [
            SimlFileExtType.PTH,
            SimlFileExtType.PTHENC
        ]
        for ext in extensions:
            if path.name.endswith(ext.value):
                return ext

        raise NotImplementedError(
            f"Unknown file extension. {path}."
            ".pth or .pth.enc is allowed"
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self._path}"

    def _check_format(self):
        if not self._path.name.startswith("snapshot_epoch_"):
            raise ValueError(
                "File name does not start with 'snapshot_epoch_': "
                f"{self._path.name}"
            )

    @property
    def file_path(self) -> pathlib.Path:
        return self._path

    @property
    def epoch(self) -> int:
        self._check_format()

        num_epoch = re.search(
            r'snapshot_epoch_(\d+)', self._path.name
        ).groups()[0]
        return int(num_epoch)

    @property
    def is_encrypted(self) -> bool:
        return self._ext_type == SimlFileExtType.PTHENC

    def load(self, device: str, *, decrypt_key: bytes = None) -> Any:
        if self.is_encrypted:
            return self._load_encrypted(
                device=device,
                decrypt_key=decrypt_key
            )
        else:
            return self._load(device=device)

    def _load(self, device: str) -> Any:
        return torch.load(self._path, map_location=device)

    def _load_encrypted(
        self,
        device: str,
        decrypt_key: bytes = None
    ) -> dict:
        if decrypt_key is None:
            raise ValueError('Feed key to load encrypted model')

        checkpoint = torch.load(
            util.decrypt_file(decrypt_key, self._path),
            map_location=device
        )
        return checkpoint

    def save(
        self,
        dump_data: object,
        overwrite: bool = False,
        encrypt_key: bytes = None
    ):
        if not overwrite:
            if self.file_path.exists():
                raise FileExistsError(
                    f"{self.file_path} already exists"
                )

        if self.is_encrypted:
            self._save_encrypted(
                dump_data=dump_data,
                key=encrypt_key
            )
        else:
            self._save(dump_data)

    def _save_encrypted(
        self,
        dump_data: object,
        key: bytes
    ) -> None:
        if key is None:
            raise ValueError(
                f"key is empty when encrpting file: {self.file_path}"
            )
        buffer = io.BytesIO()
        torch.save(dump_data, buffer)
        util.encrypt_file(key, self.file_path, buffer)

    def _save(
        self,
        dump_data: object
    ) -> None:
        torch.save(dump_data, self.file_path)

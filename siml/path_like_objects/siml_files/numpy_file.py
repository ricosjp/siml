import pathlib

import scipy.sparse as sp
import numpy as np

from siml.base.siml_typing import ArrayDataType
from siml.base.siml_enums import SimlFileExtType
from siml import util

from .interface import ISimlNumpyFile


class SimlNumpyFile(ISimlNumpyFile):
    def __init__(self, path: pathlib.Path) -> None:
        ext = self._check_extension_type(path)
        self._path = path
        self._ext_type = ext

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self._path}"

    def _check_extension_type(self, path: pathlib.Path) -> SimlFileExtType:
        extensions = [
            SimlFileExtType.NPY,
            SimlFileExtType.NPYENC,
            SimlFileExtType.NPZ,
            SimlFileExtType.NPZENC
        ]
        for ext in extensions:
            if path.name.endswith(ext.value):
                return ext

        raise NotImplementedError(
            f"Unknown file extension: {path}"
        )

    @property
    def is_encrypted(self) -> bool:
        if self._ext_type == SimlFileExtType.NPYENC:
            return True
        if self._ext_type == SimlFileExtType.NPZENC:
            return True

        return False

    @property
    def file_extension(self):
        return self._ext_type.value

    @property
    def file_path(self) -> pathlib.Path:
        return self._path

    def load(
        self,
        *,
        check_nan: bool = False,
        decrypt_key: bytes = None
    ) -> np.ndarray:
        loaded_data = self._load(decrypt_key=decrypt_key)
        if check_nan and np.any(np.isnan(loaded_data)):
            raise ValueError(
                f"NaN found in {self._path}")

        return loaded_data

    def _load(
        self,
        *,
        decrypt_key: bytes = None
    ) -> ArrayDataType:
        if self._ext_type == SimlFileExtType.NPY:
            return self._load_npy()
        if self._ext_type == SimlFileExtType.NPYENC:
            return self._load_npy_enc(decrypt_key)
        if self._ext_type == SimlFileExtType.NPZ:
            return self._load_npz()
        if self._ext_type == SimlFileExtType.NPZENC:
            return self._load_npz_enc(decrypt_key)
        raise NotImplementedError(
            "Loading function for this file extenstion is not implemented: "
            f"{self._path}"
        )

    def _load_npy(self):
        return np.load(self._path)

    def _load_npz(self):
        return sp.load_npz(self._path)

    def _load_npy_enc(self, decrypt_key: bytes):
        if decrypt_key is None:
            raise ValueError(
                "Key is None. Cannot decrypt encrypted file."
            )

        return np.load(
            util.decrypt_file(decrypt_key, self._path)
        )

    def _load_npz_enc(self, decrypt_key: bytes):
        if decrypt_key is None:
            raise ValueError(
                "Key is None. Cannot decrypt encrypted file."
            )

        return sp.load_npz(
            util.decrypt_file(decrypt_key, self._path)
        )

    def save(
        self,
        data: ArrayDataType,
        *,
        encrypt_key: bytes = None,
        overwrite: bool = True
    ) -> None:
        if not overwrite:
            if self.file_path.exists():
                raise FileExistsError(
                    f"{self._path} already exists"
                )

        if self.is_encrypted:
            if encrypt_key is None:
                raise ValueError(
                    f"key is empty when encrpting file: {self._path}"
                )

        file_basename = self._path.name.removesuffix(
            self._ext_type.value
        )
        util.save_variable(
            self._path.parent,
            file_basename=file_basename,
            data=data,
            encrypt_key=encrypt_key
        )

import pathlib
import io
import yaml

from siml.base.siml_enums import SimlFileExtType
from siml import util

from .interface import ISimlYamlFile


class SimlYamlFile(ISimlYamlFile):
    def __init__(self, path: pathlib.Path) -> None:
        ext = self._check_extension_type(path)
        self._path = path
        self._ext_type = ext

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self._path}"

    def _check_extension_type(self, path: pathlib.Path) -> SimlFileExtType:
        extensions = [
            SimlFileExtType.YAML,
            SimlFileExtType.YAMLENC,
            SimlFileExtType.YML,
            SimlFileExtType.YMLENC
        ]
        for ext in extensions:
            if path.name.endswith(ext.value):
                return ext

        raise NotImplementedError(
            f"Unknown file extension: {path}"
        )

    @property
    def is_encrypted(self) -> bool:
        return self._ext_type in [
            SimlFileExtType.YAMLENC, SimlFileExtType.YMLENC
        ]

    @property
    def file_path(self) -> pathlib.Path:
        return self._path

    def load(
        self,
        *,
        decrypt_key: bytes = None
    ) -> dict:
        if self.is_encrypted:
            return self._load_encrypted(decrypt_key=decrypt_key)
        else:
            return self._load()

    def _load(self) -> dict:
        parameters = util.load_yaml(self._path)
        return parameters

    def _load_encrypted(self, decrypt_key: bytes) -> dict:
        if decrypt_key is None:
            raise ValueError(
                "Key is None. Cannot decrypt encrypted file."
            )

        parameters = util.load_yaml(
            util.decrypt_file(decrypt_key, self._path, return_stringio=True)
        )
        return parameters

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
        data_string: str = yaml.dump(dump_data, None)
        bio = io.BytesIO(data_string.encode('utf-8'))
        util.encrypt_file(key, self.file_path, bio)

    def _save(
        self,
        dump_data: object
    ) -> None:
        with open(self.file_path, 'w') as fw:
            yaml.dump(dump_data, fw)

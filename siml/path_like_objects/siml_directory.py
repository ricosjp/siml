import pathlib
from typing import Union, Callable, Any

from siml.base.siml_enums import SimlFileExtType

from .siml_file_builder import (
    ISimlNumpyFile, ISimlYamlFile, ISimlPickleFile, SimlFileBuilder
)


class SimlDirectory:
    def __init__(self, path: pathlib.Path) -> None:
        self._path = path

    def __str__(self) -> str:
        return f"SimlDirectory: {self._path}"

    @property
    def path(self) -> pathlib.Path:
        return self._path

    def find_pickle_file(
        self,
        file_base_name: str,
        *,
        allow_missing: bool = False
    ) -> Union[ISimlPickleFile, None]:
        extensions = [
            SimlFileExtType.PKL,
            SimlFileExtType.PKLENC
        ]

        return self._find_file(
            file_base_name=file_base_name,
            extensions=extensions,
            builder=SimlFileBuilder.pickle_file,
            allow_missing=allow_missing
        )

    def find_yaml_file(
        self,
        file_base_name: str,
        *,
        allow_missing: bool = False
    ) -> Union[ISimlYamlFile, None]:
        extensions = [
            SimlFileExtType.YAML,
            SimlFileExtType.YAMLENC,
            SimlFileExtType.YML,
            SimlFileExtType.YMLENC
        ]

        return self._find_file(
            file_base_name=file_base_name,
            extensions=extensions,
            builder=SimlFileBuilder.yaml_file,
            allow_missing=allow_missing
        )

    def find_variable_file(
        self,
        variable_name: str,
        *,
        allow_missing: bool = False
    ) -> Union[ISimlNumpyFile, None]:

        extensions = [
            SimlFileExtType.NPY,
            SimlFileExtType.NPYENC,
            SimlFileExtType.NPZ,
            SimlFileExtType.NPZENC
        ]
        return self._find_file(
            variable_name,
            extensions,
            SimlFileBuilder.numpy_file,
            allow_missing=allow_missing
        )

    def _find_file(
        self,
        file_base_name: str,
        extensions: list[SimlFileExtType],
        builder: Callable[[pathlib.Path], Any],
        *,
        allow_missing: bool = False
    ):
        for ext in extensions:
            path: pathlib.Path = (self._path / (file_base_name + ext.value))
            if path.exists():
                return builder(path)

        if allow_missing:
            return None

        raise ValueError(
            f"Unknown extension or file not found for {file_base_name}"
        )

    def exist_variable_file(self, variable_name: str) -> bool:
        _file = self.find_variable_file(variable_name, allow_missing=True)
        if _file is None:
            return False
        else:
            return True

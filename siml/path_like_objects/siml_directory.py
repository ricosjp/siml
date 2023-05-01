import pathlib
from typing import Union

from siml.base.siml_enums import SimlFileExtType

from .siml_file_builder import ISimlNumpyFile, SimlFileBulider


class SimlDirectory:
    def __init__(self, path: pathlib.Path) -> None:
        self._path = path

    def __str__(self) -> str:
        return f"SimlDirectory: {self._path}"

    @property
    def path(self) -> pathlib.Path:
        return self._path

    def find_variable_file(
        self,
        variable_name: str,
        *,
        allow_missing: bool = False
    ) -> Union[ISimlNumpyFile, None]:

        extenstions = [
            SimlFileExtType.NPY.value,
            SimlFileExtType.NPYENC.value,
            SimlFileExtType.NPZ.value,
            SimlFileExtType.NPZENC.value
        ]
        for ext in extenstions:
            path = (self._path / (variable_name + ext))
            if path.exists():
                return SimlFileBulider.numpy_file(path)

        if allow_missing:
            return None

        raise ValueError(
            f"Unknown extension or file not found for {variable_name}"
        )

    def exist_variable_file(self, variable_name: str) -> bool:
        _file = self.find_variable_file(variable_name, allow_missing=True)

        if _file is None:
            return False
        else:
            return True

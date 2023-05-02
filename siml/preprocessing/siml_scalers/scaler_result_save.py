import abc
import pathlib
from typing import Optional

import numpy as np

from siml import util
from siml.siml_variables import ArrayDataType


class IScalingSaveFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(
        self,
        output_directory: pathlib.Path,
        file_basename: str,
        data: ArrayDataType,
        *,
        dtype: type = np.float32,
        encrypt_key: Optional[bytes] = None
    ) -> None:
        raise NotImplementedError()


class DefaultSaveFunction(IScalingSaveFunction):
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        output_directory: pathlib.Path,
        file_basename: str,
        data: ArrayDataType,
        *,
        dtype: type = np.float32,
        encrypt_key: Optional[bytes] = None
    ) -> None:
        util.save_variable(
            output_directory,
            file_basename=file_basename,
            data=data,
            dtype=dtype,
            encrypt_key=encrypt_key
        )

import datetime as dt
import gc
from typing import Optional

import numpy as np
import scipy.sparse as sp

from siml.path_like_objects import ISimlFile
from siml.preprocessing.siml_scalers import (
    scale_functions, scale_variables, ISimlScaler)


class SimlScalerWrapper(ISimlScaler):

    def __init__(
        self,
        setting_data: str,
        *,
        componentwise: bool = True,
        key: Optional[bytes] = None,
        group_id: Optional[int] = None,
        **kwards
    ):

        self.setting_data = setting_data
        self.converter = scale_functions.create_scaler(
            setting_data,
            **kwards
        )
        self.key = key
        self.componentwise = componentwise
        self.group_id = group_id

    @property
    def use_diagonal(self) -> bool:
        return self.converter.use_diagonal

    def is_erroneous(self) -> bool:
        return self.converter.is_erroneous()

    def partial_fit(
        self,
        data: scale_variables.SimlScaleDataType
    ) -> None:
        wrapped_data = scale_variables.create_wrapper(data)
        reshaped_data = wrapped_data.reshape(
            componentwise=self.componentwise,
            skip_nan=True,
            use_diagonal=self.use_diagonal
        )

        self.converter.partial_fit(reshaped_data)
        return

    def transform(self, data: scale_variables.SimlScaleDataType) -> np.ndarray:
        wrapped_data = scale_variables.create_wrapper(data)
        reshaped_data = wrapped_data.reshape(
            componentwise=self.componentwise,
            skip_nan=False,
            use_diagonal=self.converter.use_diagonal
        )
        return self.converter.transform(reshaped_data)

    def inverse_transform(
        self,
        data: scale_variables.SimlScaleDataType
    ) -> scale_variables.SimlScaleDataType:
        wrapped_data = scale_variables.create_wrapper(data)
        reshaped_data = wrapped_data.reshape(
            componentwise=self.componentwise,
            skip_nan=False,
            use_diagonal=self.converter.use_diagonal
        )
        return self.converter.inverse_transform(reshaped_data)

    def lazy_partial_fit(
        self,
        data_files: list[ISimlFile]
    ) -> None:
        for data_file in data_files:
            print(f"Start load data: {data_file}")
            print(dt.datetime.now())
            data = self._load_file(data_file)
            print(f"Start partial_fit: {data_file}")
            print(dt.datetime.now())
            self.partial_fit(data)
            print(f"Start del: {data_file}")
            print(dt.datetime.now())
            del data
            print(f"Start GC: {data_file}")
            print(dt.datetime.now())
            gc.collect()
            print(f"Finish one iter: {data_file}")
            print(dt.datetime.now())
        return

    def _load_file(
        self,
        siml_file: ISimlFile
    ) -> scale_variables.SimlScaleDataType:

        loaded_data = siml_file.load(decrypt_key=self.key)

        if siml_file.get_file_extension() in [".npz.enc", ".npz"]:
            if not sp.issparse(loaded_data):
                raise ValueError(
                    f"Data type not understood for: {siml_file.file_path}"
                )

        return loaded_data

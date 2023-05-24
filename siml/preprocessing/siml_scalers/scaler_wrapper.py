import datetime as dt
import gc
from typing import Optional, Union
import warnings

import numpy as np
import scipy.sparse as sp

from siml.base.siml_enums import SimlFileExtType
from siml.path_like_objects import ISimlNumpyFile
from siml.preprocessing.siml_scalers import ISimlScaler, scale_functions
from siml.siml_variables import ArrayDataType, create_siml_arrray


class SimlScalerWrapper(ISimlScaler):
    @classmethod
    def create(cls, dict_data: dict, key: Optional[bytes] = None):

        # Pass all dict_data['preprocess_converter'] items
        # to avoid Exception by checking when initialization (Ex. IsoAMSclaer)
        _cls = cls(
            dict_data["method"],
            componentwise=dict_data.get("componentwise", True),
            key=key,
            **dict_data['preprocess_converter']
        )
        # After initialization, set other private properties such as var_, max_
        for k, v in dict_data['preprocess_converter'].items():
            setattr(_cls.converter, k, v)
        return _cls

    def __init__(
        self,
        method_name: str,
        *,
        componentwise: bool = True,
        key: Optional[bytes] = None,
        group_id: Optional[int] = None,
        **kwards
    ):

        self.method_name = method_name
        self.converter = scale_functions.create_scaler(
            method_name,
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
        data: ArrayDataType
    ) -> None:
        wrapped_data = create_siml_arrray(data)
        reshaped_data = wrapped_data.reshape(
            componentwise=self.componentwise,
            skip_nan=True,
            use_diagonal=self.use_diagonal
        )
        if reshaped_data.size == 0:
            warnings.warn(
                "Found array with 0 sample(s) after deleting nan items"
            )
            return

        self.converter.partial_fit(reshaped_data)
        return

    def transform(
        self,
        data: ArrayDataType
    ) -> Union[np.ndarray, sp.coo_matrix]:

        wrapped_data = create_siml_arrray(data)
        result = wrapped_data.apply(
            self.converter.transform,
            componentwise=self.componentwise,
            skip_nan=False,
            use_diagonal=False
        )
        return result

    def inverse_transform(
        self,
        data: ArrayDataType
    ) -> Union[np.ndarray, sp.coo_matrix]:
        wrapped_data = create_siml_arrray(data)
        result = wrapped_data.apply(
            self.converter.inverse_transform,
            componentwise=self.componentwise,
            skip_nan=False,
            use_diagonal=False
        )
        return result

    def lazy_partial_fit(
        self,
        data_files: list[ISimlNumpyFile]
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

    def get_dumped_dict(self) -> dict:
        dumped_dict = {
            'method': self.method_name,
            'componentwise': self.componentwise,
            'preprocess_converter': vars(self.converter)
        }
        return dumped_dict

    def _load_file(
        self,
        siml_file: ISimlNumpyFile
    ) -> ArrayDataType:

        loaded_data = siml_file.load(decrypt_key=self.key)

        if siml_file.file_extension in [
                SimlFileExtType.NPZENC.value, SimlFileExtType.NPZ.value]:
            if not sp.issparse(loaded_data):
                raise ValueError(
                    f"Data type not understood for: {siml_file.file_path}"
                )

        return loaded_data

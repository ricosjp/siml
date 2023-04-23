from __future__ import annotations
import multiprocessing as multi
import warnings
import pathlib
from typing import Optional, Union

from siml import setting
from siml import util
from siml.path_like_objects import ISimlFile, SimlFileBulider

from .siml_scalers import SimlScalerWrapper
from .siml_scalers.scale_variables import SimlScaleDataType


class ScalersComposition():
    @classmethod
    def create(
        cls,
        converter_parameters_pkl: pathlib.Path,
        key: Optional[bytes] = None
    ) -> ScalersComposition:
        siml_file = SimlFileBulider.create(converter_parameters_pkl)
        preprocess_setting = setting.PreprocessSetting(
            siml_file.load(decrypt_key=key)
        )
        return cls(
            preprocess_dict=preprocess_setting.preprocess,
            decrypt_key=key
        )

    def __init__(
        self,
        preprocess_dict: dict,
        max_process: Optional[int] = None,
        decrypt_key: Optional[bytes] = None
    ) -> None:

        self._setting = preprocess_dict
        self._decrypt_key = decrypt_key
        self.max_process = util.determine_max_process(max_process)
        # varaible name to saler name (variable name except same_as)
        self._variable_name_to_scaler: dict[str, str] \
            = self._init_scalers_relationship()

        # variable_name to siml_scelar
        self._scalers_dict: dict[str, SimlScalerWrapper] \
            = self._init_scalers_dict()

    def get_variable_names(self, group_id: Optional[int] = None) -> list[str]:
        variable_names = list(self._variable_name_to_scaler.keys())
        if group_id is None:
            return variable_names

        variable_names = [
            k for k in variable_names
            if self.get_scaler(k).group_id == group_id
        ]
        return variable_names

    def get_scaler_names(self, group_id: Optional[int] = None) -> list[str]:
        scaler_names = list(self._scalers_dict.keys())
        if group_id is None:
            return scaler_names

        scaler_names = [
            k for k in scaler_names
            if self._scalers_dict[k].group_id == group_id
        ]
        return scaler_names

    def get_scaler(
        self,
        variable_name: str,
        allow_missing: bool = False
    ) -> SimlScalerWrapper:
        scaler_name = self._variable_name_to_scaler.get(variable_name)
        scaler = self._scalers_dict.get(scaler_name)
        if allow_missing:
            return scaler

        if scaler is None:
            raise ValueError(
                f"No Scaler for {scaler_name} is found."
            )
        return scaler

    def load(self, preprocessor_pkl: pathlib.Path) -> None:
        siml_file = SimlFileBulider.create(preprocessor_pkl)
        assert siml_file.get_file_extension() in [".enc.pkl", ".pkl"]

        parameters: dict = siml_file.load(
            decrypt_key=self._decrypt_key
        )

        for k in self._scalers_dict.keys():
            if k in parameters.keys():
                raise ValueError(
                    "Attempted to load parameters, "
                    f"but preprocessor for {k} is not defined "
                    "in parameters file."
                )
            self._scalers_dict[k].converter = parameters[k]

    def get_dumped_object(self) -> None:
        dict_to_dump = {
            k: vars(self.get_scaler(k).converter)
            for k in self.get_scaler_names()
        }
        return dict_to_dump

    def lazy_partial_fit(
        self,
        scaler_name_to_files: dict[str, list[ISimlFile]]
    ):
        preprocessor_inputs: list[tuple[str, list[ISimlFile]]] \
            = [
                (name, files)
                for name, files in scaler_name_to_files.items()
        ]

        with multi.Pool(self.max_process) as pool:
            results = pool.starmap(
                self._lazy_partial_fit,
                preprocessor_inputs,
                chunksize=1
            )
        for name, scaler in results:
            self._scalers_dict[name] = scaler

    def transform(
        self,
        variable_name: str,
        data: SimlScaleDataType
    ) -> SimlScaleDataType:
        scaler = self.get_scaler(variable_name)
        transformed_data = scaler.transform(data)
        return transformed_data

    def transform_file(
        self,
        variable_name: str,
        siml_file: ISimlFile
    ) -> SimlScaleDataType:

        loaded_data = siml_file.load(
            decrypt_key=self._decrypt_key
        )
        scaler = self.get_scaler(variable_name)
        transformed_data = scaler.transform(loaded_data)
        return transformed_data

    def transform_dict(
        self,
        dict_data: dict[str, SimlScaleDataType]
    ) -> dict[str, SimlScaleDataType]:

        converted_dict_data: dict[str, SimlScaleDataType] = {}
        for variable_name, data in dict_data.items():
            scaler = self.get_scaler(variable_name, allow_missing=True)
            if scaler is None:
                warnings.warn(
                    f"Scaler for {variable_name} is not found. Skipped"
                )
                continue

            converted_data = scaler.transform(data)
            converted_dict_data[variable_name] = converted_data

        if len(converted_dict_data) == 0:
            raise ValueError(
                'No converted data found. '
                'Check the preprocessed directory set correctly.'
            )
        return converted_dict_data

    def inverse_transform(
        self,
        variable_name: str,
        data: SimlScaleDataType
    ):
        scaler = self.get_scaler(variable_name)
        return scaler.inverse_transform(data)

    def inverse_transform_dict(
        self,
        dict_data: dict[str, SimlScaleDataType]
    ) -> dict[str, SimlScaleDataType]:

        converted_dict_data: dict[str, SimlScaleDataType] = {}
        for variable_name, data in dict_data.items():
            scaler = self.get_scaler(variable_name, allow_missing=True)
            if scaler is None:
                warnings.warn(
                    f"Scaler for {variable_name} is not found. Skipped"
                )
                continue

            converted_data = scaler.inverse_transform(data)
            converted_dict_data[variable_name] = converted_data
        return converted_dict_data

    def _lazy_partial_fit(
        self,
        variable_name: str,
        data_files: list[ISimlFile]
    ) -> tuple[str, SimlScalerWrapper]:
        scaler = self.get_scaler(variable_name)

        scaler.lazy_partial_fit(data_files)
        return (variable_name, scaler)

    def _init_scalers_relationship(
        self
    ) -> dict[str, str]:

        _dict = {}
        for variable_name, p_setting in self._setting.items():
            parent = p_setting.get('same_as')
            if parent is None:
                _dict[variable_name] = variable_name
            else:
                _dict[variable_name] = parent
                # same_as is set so no need to prepare preprocessor
        return _dict

    def _init_scalers_dict(
        self,
        group_id: int = None
    ) -> dict[str, SimlScalerWrapper]:
        preprocessor_inputs = [
            (variable_name, preprocess_setting)
            for variable_name, preprocess_setting
            in self._setting.items()
            if group_id is None or preprocess_setting['group_id'] == group_id
        ]

        _scalers_dict: dict[str, SimlScalerWrapper] = {}
        for k, _setting in preprocessor_inputs:
            v = self._init_siml_scaler(_setting)
            if v is None:
                continue

            _scalers_dict[k] = v

        return _scalers_dict

    def _init_siml_scaler(
        self,
        preprocess_setting: dict
    ) -> Union[SimlScalerWrapper, None]:

        if preprocess_setting.get('same_as') is not None:
            # same_as is set so no need to prepare preprocessor
            return None

        preprocess_converter = SimlScalerWrapper(
            preprocess_setting["method"],
            key=self._decrypt_key,
            componentwise=preprocess_setting['componentwise'],
            power=preprocess_setting.get('power', 1.),
            other_components=preprocess_setting['other_components']
        )

        return preprocess_converter

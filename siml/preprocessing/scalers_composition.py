from __future__ import annotations
import multiprocessing as multi
import warnings
import pathlib
from typing import Optional, Union, Final

from siml import util
from siml.path_like_objects import SimlFileBuilder, ISimlNumpyFile
from siml.siml_variables import ArrayDataType

from .siml_scalers import SimlScalerWrapper


class ScalersComposition():
    REGISTERED_KEY: Final[str] = "variable_name_to_scalers"

    @classmethod
    def create_from_file(
        cls,
        converter_parameters_pkl: pathlib.Path,
        max_process: Optional[int] = None,
        key: Optional[bytes] = None
    ) -> ScalersComposition:
        siml_file = SimlFileBuilder.pickle_file(converter_parameters_pkl)
        parameters: dict = siml_file.load(
            decrypt_key=key
        )
        variable_name_to_scalers = parameters.pop(
            cls.REGISTERED_KEY, None
        )
        scalers_dict = ScalersComposition._load_scalers(parameters, key=key)
        if variable_name_to_scalers is None:
            # When old version, key "varaible_name_to_scalers" does not exist
            variable_name_to_scalers = {k: k for k in scalers_dict.keys()}

        return cls(
            variable_name_to_scalers=variable_name_to_scalers,
            scalers_dict=scalers_dict,
            max_process=max_process,
            decrypt_key=key
        )

    @classmethod
    def create_from_dict(
        cls,
        preprocess_dict: dict,
        max_process: Optional[int] = None,
        key: Optional[bytes] = None
    ) -> ScalersComposition:
        variable_name_to_scalers = \
            ScalersComposition._init_scalers_relationship(preprocess_dict)
        scalers_dict = ScalersComposition._init_scalers_dict(
            preprocess_dict,
            key=key
        )
        return cls(
            variable_name_to_scalers=variable_name_to_scalers,
            scalers_dict=scalers_dict,
            max_process=max_process,
            decrypt_key=key
        )

    def __init__(
        self,
        variable_name_to_scalers: dict[str, str],
        scalers_dict: dict[str, SimlScalerWrapper],
        max_process: Optional[int] = None,
        decrypt_key: Optional[bytes] = None
    ) -> None:

        self._decrypt_key = decrypt_key
        self.max_process = util.determine_max_process(max_process)
        # varaible name to saler name
        # scaler name corresponds with variable name except same_as
        self._variable_name_to_scaler = variable_name_to_scalers

        # variable_name to siml_scelar
        self._scalers_dict = scalers_dict

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

    def get_dumped_object(self) -> dict:
        dumped_dict = {
            k: self.get_scaler(k).get_dumped_dict()
            for k in self.get_scaler_names()
        }
        # add relationship
        dumped_dict[self.REGISTERED_KEY] \
            = self._variable_name_to_scaler
        return dumped_dict

    def lazy_partial_fit(
        self,
        scaler_name_to_files: dict[str, list[ISimlNumpyFile]]
    ) -> None:
        preprocessor_inputs: list[tuple[str, list[ISimlNumpyFile]]] \
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
        data: ArrayDataType
    ) -> ArrayDataType:
        scaler = self.get_scaler(variable_name)
        transformed_data = scaler.transform(data)
        return transformed_data

    def transform_file(
        self,
        variable_name: str,
        siml_file: ISimlNumpyFile
    ) -> ArrayDataType:

        loaded_data = siml_file.load(
            decrypt_key=self._decrypt_key
        )
        scaler = self.get_scaler(variable_name)
        transformed_data = scaler.transform(loaded_data)
        return transformed_data

    def transform_dict(
        self,
        dict_data: dict[str, ArrayDataType]
    ) -> dict[str, ArrayDataType]:

        converted_dict_data: dict[str, ArrayDataType] = {}
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
        data: ArrayDataType
    ) -> ArrayDataType:
        scaler = self.get_scaler(variable_name)
        return scaler.inverse_transform(data)

    def inverse_transform_dict(
        self,
        dict_data: dict[str, ArrayDataType]
    ) -> dict[str, ArrayDataType]:

        converted_dict_data: dict[str, ArrayDataType] = {}
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
        data_files: list[ISimlNumpyFile]
    ) -> tuple[str, SimlScalerWrapper]:
        scaler = self.get_scaler(variable_name)

        scaler.lazy_partial_fit(data_files)
        return (variable_name, scaler)

    @staticmethod
    def _init_scalers_relationship(
        setting_dict: dict
    ) -> dict[str, str]:

        _dict = {}
        for variable_name, p_setting in setting_dict.items():
            parent = p_setting.get('same_as')
            if parent is None:
                _dict[variable_name] = variable_name
            else:
                _dict[variable_name] = parent
                # same_as is set so no need to prepare preprocessor
        return _dict

    @staticmethod
    def _init_scalers_dict(
        setting_dict: dict,
        key: Optional[bytes] = None,
        group_id: Optional[int] = None
    ) -> dict[str, SimlScalerWrapper]:
        preprocessor_inputs = [
            (variable_name, preprocess_setting)
            for variable_name, preprocess_setting
            in setting_dict.items()
            if group_id is None or preprocess_setting['group_id'] == group_id
        ]

        _scalers_dict: dict[str, SimlScalerWrapper] = {}
        for k, _setting in preprocessor_inputs:
            v = ScalersComposition._init_siml_scaler(_setting, key=key)
            if v is None:
                continue

            _scalers_dict[k] = v

        return _scalers_dict

    @staticmethod
    def _init_siml_scaler(
        preprocess_setting: dict,
        key: Optional[bytes] = None
    ) -> Union[SimlScalerWrapper, None]:

        if preprocess_setting.get('same_as') is not None:
            # same_as is set so no need to prepare preprocessor
            return None

        preprocess_converter = SimlScalerWrapper(
            preprocess_setting["method"],
            key=key,
            componentwise=preprocess_setting['componentwise'],
            power=preprocess_setting.get('power', 1.),
            other_components=preprocess_setting['other_components']
        )

        return preprocess_converter

    @staticmethod
    def _load_scalers(
        parameters: dict,
        key: Optional[bytes] = None
    ) -> dict[str, SimlScalerWrapper]:
        scalers_dict = {}
        for k, dict_data in parameters.items():
            scalers_dict[k] = SimlScalerWrapper.create(
                dict_data,
                key=key
            )
        return scalers_dict

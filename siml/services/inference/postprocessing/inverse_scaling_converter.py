from __future__ import annotations
from typing import Optional, Union
import pathlib

from siml.preprocessing import ScalersComposition
from siml.base.siml_typing import ArrayDataType


class InverseScalingConverter:
    @classmethod
    def create(
        cls,
        converter_parameters_pkl: pathlib.Path,
        key: bytes = None
    ) -> InverseScalingConverter:
        converters = ScalersComposition.create_from_file(
            converter_parameters_pkl=converter_parameters_pkl,
            key=key
        )
        return cls(converters)

    def __init__(self, converters: ScalersComposition):
        self._converters = converters

    def inverse_scaling(
        self,
        dict_data_x: dict,
        dict_data_y: dict,
        *,
        dict_data_y_answer: Optional[dict] = None
    ) -> tuple[
            dict[str, ArrayDataType],
            dict[str, ArrayDataType],
            dict[str, ArrayDataType]
    ]:
        _dict_data_x = self._format_dict_shape(dict_data_x)
        _dict_data_y = self._format_dict_shape(dict_data_y)
        _dict_data_y_answer = self._format_dict_shape(dict_data_y_answer)

        _dict_data_x = self._inverse_process(_dict_data_x)
        _dict_data_y = self._inverse_process(_dict_data_y)
        _dict_data_y_answer = self._inverse_process(_dict_data_y_answer)
        return _dict_data_x, _dict_data_y, _dict_data_y_answer

    def _format_dict_shape(
        self,
        dict_data: Union[dict, None]
    ) -> Union[dict[str, ArrayDataType], None]:
        if dict_data is None:
            return None

        if len(dict_data) == 0:
            return None

        if isinstance(list(dict_data.values())[0], dict):
            # REVIEW: Maybe, not appropriate to overwrite value
            #  for variable name
            return_dict_data = {
                variable_name: data
                for value in dict_data.values()
                for variable_name, data in value.items()
                if variable_name in self._converters.get_variable_names()
            }
        else:
            return_dict_data = {
                variable_name: data
                for variable_name, data in dict_data.items()
                if variable_name in self._converters.get_variable_names()
            }
        return return_dict_data

    def _inverse_process(
        self,
        dict_data: Union[dict[str, ArrayDataType], None]
    ) -> dict[str, ArrayDataType]:
        if dict_data is None:
            return {}

        dict_data_answer = self._converters.inverse_transform_dict(dict_data)
        return dict_data_answer

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
        scalers = ScalersComposition.create_from_file(
            converter_parameters_pkl=converter_parameters_pkl,
            key=key
        )
        return cls(scalers)

    def __init__(self, scalers: ScalersComposition):
        self._scalers = scalers

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
        _dict_data_x = self._inverse_process(
            self._select_reversible_variables(dict_data_x)
        )
        _dict_data_y = self._inverse_process(
            self._select_reversible_variables(dict_data_y)
        )
        _dict_data_y_answer = self._inverse_process(
            self._select_reversible_variables(dict_data_y_answer)
        )
        return _dict_data_x, _dict_data_y, _dict_data_y_answer

    def _inverse_process(
        self,
        dict_data: Union[dict[str, ArrayDataType], None]
    ) -> dict[str, ArrayDataType]:
        if dict_data is None:
            return {}

        dict_data_answer = self._scalers.inverse_transform_dict(dict_data)
        return dict_data_answer

    def _select_reversible_variables(
        self,
        dict_data: Union[dict, None]
    ) -> Union[dict[str, ArrayDataType], None]:
        if dict_data is None:
            return None

        if len(dict_data) == 0:
            return None

        return_dict_data = {
            name: value
            for name, value in dict_data.items()
            if name in self._scalers.get_variable_names()
        }

        return return_dict_data

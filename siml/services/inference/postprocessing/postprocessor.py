from typing import Optional, Union

import numpy as np

from siml.preprocessing import ScalersComposition
from siml.services.path_rules import SimlPathRules
from siml.setting import CollectionVariableSetting, TrainerSetting
from siml.services.inference.record_object import (
    PostPredictionRecord, PredictionRecord
)

from .inverse_scaling_converter import InverseScalingConverter


class PostProcessor:
    def __init__(
        self,
        trainer_setting: TrainerSetting,
        perform_inverse: bool,
        *,
        scalers: Optional[ScalersComposition],
    ) -> None:
        self._trainer_setting = trainer_setting
        self._perform_inverse = perform_inverse
        self._path_rules = SimlPathRules()
        if self._perform_inverse:
            self._converter = InverseScalingConverter(scalers)

    def convert(
        self,
        record: PredictionRecord,
        start_datetime: str
    ) -> PostPredictionRecord:
        dict_var_x = self._separate_data(
            record.x.to_numpy(),
            self._trainer_setting.inputs
        )
        dict_var_y = self._separate_data(
            record.y.to_numpy(),
            self._trainer_setting.outputs
        )
        dict_var_y_pred = self._separate_data(
            record.y_pred.to_numpy(),
            self._trainer_setting.outputs
        )
        if not self._perform_inverse:
            return dict_var_x, dict_var_y_pred, dict_var_y

        inversed_dict_x, inversed_dict_y, inversed_dict_answer = \
            self._converter.inverse_scaling(
                dict_var_x,
                dict_var_y_pred,
                dict_data_y_answer=dict_var_y
            )

        converted_record = PostPredictionRecord(
            dict_x=inversed_dict_x,
            dict_y=inversed_dict_y,
            dict_answer=inversed_dict_answer,
            original_shapes=record.original_shapes,
            data_directory=record.data_directory,
            inference_time=record.inference_time,
            inference_start_datetime=start_datetime
        )
        return converted_record

    def _separate_data(
        self,
        data: Union[list, dict],
        descriptions: CollectionVariableSetting,
        *,
        axis=-1
    ) -> dict:
        # TODO
        # this function is not implemented here.
        # cohesion decreases.

        if isinstance(data, dict):
            return {
                key: self._separate_list_data(
                    data[key], descriptions.variables[key],
                    axis=axis
                )
                for key in data.keys()
            }
        elif isinstance(data, list):
            return self._separate_list_data(data, descriptions, axis=axis)

        else:
            raise NotImplementedError(
                f"Unknown data type: {type(data)}"
            )

    def _separate_list_data(
        self,
        data: list,
        descriptions: CollectionVariableSetting,
        *,
        axis=-1
    ) -> dict[str, np.ndarray]:

        # TODO
        # this function is not implemented here.
        # cohesion decreases.
        if len(data) == 0:
            return {}

        data_dict = {}
        index = 0
        data = np.swapaxes(data, 0, axis)
        for description in descriptions.variables:
            dim = description.dim
            data_dict.update(
                {
                    description.name:
                    np.swapaxes(data[index:index+dim], 0, axis)
                }
            )
            index += dim
        return data_dict

from __future__ import annotations

import pathlib
from typing import Optional, Union

import numpy as np

from siml.preprocessing import ScalersComposition
from siml.services.inference import InnerInfererSetting
from siml.services.inference.record_object import (
    PostPredictionRecord,
    PredictionRecord
)
from siml.services.path_rules import SimlPathRules
from siml.setting import CollectionVariableSetting

from .inverse_scaling_converter import InverseScalingConverter
from .post_fem_data import PostFEMDataConverter


class PostProcessor:
    def __init__(
        self,
        inner_setting: InnerInfererSetting,
        fem_data_converter: PostFEMDataConverter,
        *,
        scalers: Optional[ScalersComposition] = None,
    ) -> None:
        self._inner_setting = inner_setting
        self._trainer_setting = inner_setting.trainer_setting
        self._perform_inverse = inner_setting.perform_inverse
        self._fem_data_converter = fem_data_converter
        self._path_rules = SimlPathRules()
        if self._perform_inverse:
            if scalers is None:
                raise ValueError(
                    "scalers is None. When perform inverse, "
                    "scalers must be set."
                )
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

        write_simulation_case_dir = \
            self._inner_setting.get_write_simulation_case_dir(
                record.data_directory
            )
        fem_data = None
        if not self._is_skip_fem_data(write_simulation_case_dir):
            fem_data = self._fem_data_converter.create(
                record, write_simulation_case_dir
            )

        converted_record = PostPredictionRecord(
            dict_x=inversed_dict_x,
            dict_y=inversed_dict_y,
            dict_answer=inversed_dict_answer,
            original_shapes=record.original_shapes,
            data_directory=record.data_directory,
            inference_time=record.inference_time,
            inference_start_datetime=start_datetime,
            fem_data=fem_data
        )
        return converted_record

    def _separate_data(
        self,
        data: Union[list, dict, np.ndarray],
        descriptions: CollectionVariableSetting,
        *,
        axis=-1
    ) -> dict:
        # TODO
        # this function may not be implemented here.

        if isinstance(data, dict):
            return {
                key:
                self._separate_data(
                    data[key], descriptions.variables[key], axis=axis)
                for key in data.keys()
            }
        if len(data) == 0:
            return {}

        data_dict = {}
        index = 0
        data = np.swapaxes(data, 0, axis)
        for description in descriptions.variables:
            dim = description.dim
            data_dict.update({
                description.name:
                np.swapaxes(data[index:index+dim], 0, axis)})
            index += dim
        return data_dict

    def _is_skip_fem_data(
        self,
        write_simulation_base: Optional[pathlib.Path] = None
    ) -> bool:
        if self._inner_setting.inferer_setting.skip_fem_data_creation:
            return True

        if write_simulation_base is None:
            return True

        if not write_simulation_base.exists():
            print(
                f"{write_simulation_base} does not exist."
                "Thus, skip creating fem data."
            )
            return True

        return False

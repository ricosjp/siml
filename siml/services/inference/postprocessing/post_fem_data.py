import abc
import pathlib
from typing import Optional, Union

import femio
import numpy as np

from siml import util, setting
from siml.preprocessing.converter import ILoadFunction
from siml.utils import fem_data_utils


class IFEMDataAdditionFunction(metaclass=abc.ABCMeta):
    def __call__(
        self,
        fem_data: femio.FEMData,
        write_simulation_base: pathlib.Path
    ) -> None:
        """
        Register Additional values to fem data.
        This is a destructive function.

        Parameters
        ----------
        fem_data : femio.FEMData
            fem data
        write_simulation_base : pathlib.Path
            Path to directory of simulation file
        """
        raise NotImplementedError()


class PostFEMDataConverter:
    def __init__(
        self,
        inferer_setting: setting.InfererSetting,
        conversion_setting: setting.ConversionSetting,
        load_function: Optional[ILoadFunction] = None,
        data_addition_function: Optional[IFEMDataAdditionFunction] = None,
    ) -> None:
        self._inferer_setting = inferer_setting
        self._conversion_setting = conversion_setting
        self._load_function = load_function
        self._add_data_function = data_addition_function

    def create(
        self,
        dict_data_x: dict[str, np.ndarray],
        dict_data_y: dict[str, np.ndarray],
        dict_data_answer: Optional[dict[str, np.ndarray]] = None,
        write_simulation_case_dir: Union[pathlib.Path, None] = None,
        base_fem_data: Optional[femio.FEMData] = None
    ) -> Union[femio.FEMData, None]:

        if base_fem_data is not None:
            fem_data = base_fem_data
        else:
            fem_data = self._create_fem_data(write_simulation_case_dir)

        if fem_data is None:
            return None

        if self._inferer_setting.convert_to_order1:
            fem_data = fem_data.to_first_order()

        self._add_fem_data(
            fem_data,
            dict_data_x=dict_data_x,
            dict_data_y=dict_data_y,
            dict_data_answer=dict_data_answer,
            write_simulation_case_dir=write_simulation_case_dir
        )

        return fem_data

    def _create_fem_data(
        self,
        write_simulation_case_dir: Union[pathlib.Path, None] = None
    ) -> Union[femio.FEMData, None]:

        try:
            fem_data = self._create_simulation_fem_data(
                write_simulation_case_dir
            )
            return fem_data
        except ValueError as e:
            write_simulation_stem = self._inferer_setting.write_simulation_stem
            read_simulation_type = self._inferer_setting.read_simulation_type
            print(
                f"{e}\n"
                'Could not read FEMData object, set None\n'
                f"write_simulation_case_dir: {write_simulation_case_dir}\n"
                f"write_simulation_stem: {write_simulation_stem}\n"
                f"read_simulation_type: {read_simulation_type}\n"
            )
            return None

    def _create_simulation_fem_data(
        self,
        write_simulation_case_dir: pathlib.Path
    ) -> femio.FEMData:

        write_simulation_stem = self._inferer_setting.write_simulation_stem
        read_simulation_type = self._inferer_setting.read_simulation_type

        if not self._conversion_setting.skip_femio:
            fem_data = femio.FEMData.read_directory(
                read_simulation_type,
                write_simulation_case_dir,
                stem=write_simulation_stem,
                save=False,
                read_mesh_only=False
            )
            return fem_data

        if self._load_function is not None:
            required_file_names = self._conversion_setting.required_file_names
            if len(required_file_names) == 0:
                raise ValueError(
                    'Please specify required_file_names when skip_femio '
                    'is True.'
                )

            data_files = util.collect_files(
                write_simulation_case_dir, required_file_names)
            data_dict, fem_data = self._load_function(
                data_files, write_simulation_case_dir
            )
            wrapped_fem_data = fem_data_utils.FemDataWrapper(fem_data)
            wrapped_fem_data.update_fem_data(data_dict, allow_overwrite=True)
            fem_data = wrapped_fem_data.fem_data
            return fem_data

        raise ValueError(
            'When skip_femio is True, please feed load_function.'
        )

    def _add_fem_data(
        self,
        fem_data: femio.FEMData,
        dict_data_x: dict[str, np.ndarray],
        dict_data_y: dict[str, np.ndarray],
        dict_data_answer: Optional[dict[str, np.ndarray]] = None,
        write_simulation_case_dir: Optional[pathlib.Path] = None
    ) -> None:
        wrapped_fem_data = fem_data_utils.FemDataWrapper(fem_data)
        wrapped_fem_data.update_fem_data(dict_data_x, prefix='input_')
        if dict_data_answer is not None:
            wrapped_fem_data.update_fem_data(
                dict_data_answer, prefix='answer_'
            )
        wrapped_fem_data.update_fem_data(dict_data_y, prefix='predicted_')
        wrapped_fem_data.add_difference(
            dict_data_y, dict_data_answer, prefix='difference_'
        )
        wrapped_fem_data.add_abs_difference(
            dict_data_y, dict_data_answer, prefix='difference_abs_'
        )

        fem_data = wrapped_fem_data.fem_data
        if self._add_data_function is not None:
            self._add_data_function(fem_data, write_simulation_case_dir)

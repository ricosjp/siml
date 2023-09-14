import abc
import pathlib
from typing import Optional

import femio
import numpy as np
import pandas as pd

from siml import setting
from siml.services.inference.inner_setting import InnerInfererSetting
from siml.services.inference.record_object import PostPredictionRecord


class IInfererSaveFunction(metaclass=abc.ABCMeta):
    def __call__(
        self,
        output_directory: pathlib.Path,
        fem_data: femio.FEMData,
        overwrite: bool,
        write_simulation_type: str,
        **kwards
    ) -> None:
        raise NotImplementedError()


class SaveProcessor():
    def __init__(
        self,
        inner_setting: InnerInfererSetting,
        user_save_function: Optional[IInfererSaveFunction] = None,
    ) -> None:
        self._inner_setting = inner_setting
        self._inferer_setting = inner_setting.inferer_setting
        self._conversion_setting = inner_setting.conversion_setting
        self._user_save_function = user_save_function

    def run(
        self,
        records: list[PostPredictionRecord],
        *,
        save_summary: bool = True
    ) -> None:

        # Save each results
        self._save_each_results(
            records,
            save_x=save_summary
        )

        if save_summary:
            self._save_summary(records)

    def _save_summary(self, records: list[PostPredictionRecord]):
        # Save overall settings
        output_directory = self._inner_setting.get_output_directory(
            date_string=records[0].inference_start_datetime
        )
        self._save_settings(output_directory)

        self._save_logs(
            records=records,
            output_directory=output_directory,
        )

    def _save_each_results(
        self,
        records: list[PostPredictionRecord],
        save_x: bool = False
    ) -> None:
        for record in records:
            output_directory = self._inner_setting.get_output_directory(
                record.inference_start_datetime,
                data_directory=record.data_directory,
            )

            if save_x:
                self._save_npy_data(record.dict_x, output_directory)
            self._save_npy_data(record.dict_y, output_directory)

            if record.fem_data is None:
                continue

            self._save_fem_data(
                output_directory=output_directory,
                fem_data=record.fem_data
            )

    def _save_fem_data(
        self,
        output_directory: pathlib.Path,
        fem_data: femio.FEMData
    ):
        default_save_func = FEMDataSaveFunction()
        overwrite = self._inferer_setting.overwrite
        write_simulation_type = self._inferer_setting.write_simulation_type

        default_save_func(
            output_directory=output_directory,
            fem_data=fem_data,
            overwrite=overwrite,
            write_simulation_type=write_simulation_type,
            less_output=self._inferer_setting.less_output
        )
        if self._user_save_function is None:
            return

        self._user_save_function(
            output_directory,
            fem_data,
            overwrite=overwrite,
            write_simulation_type=write_simulation_type,
        )

    def _save_npy_data(
        self,
        data_dict: dict[str, np.ndarray],
        output_directory: pathlib.Path
    ) -> None:
        if not output_directory.exists():
            output_directory.mkdir(parents=True, exist_ok=True)

        for variable_name, data in data_dict.items():
            np.save(output_directory / f"{variable_name}.npy", data)
        return

    def _save_settings(self, output_directory: pathlib.Path) -> None:
        """Save inference results information.

        Parameters
        ----------
        results: Dict
            Inference results.
        """
        output_directory.mkdir(parents=True, exist_ok=True)
        setting.write_yaml(
            self._inner_setting.main_setting,
            output_directory / 'settings.yml',
            key=self._inner_setting.main_setting.get_crypt_key()
        )
        return

    def _save_logs(
        self,
        records: list[PostPredictionRecord],
        output_directory: pathlib.Path
    ) -> None:
        column_names = [
            'loss', 'raw_loss', 'output_directory', 'data_directory',
            'inference_time'
        ]

        log_dict = {
            column_name: [
                getattr(rec, column_name) for rec in records
            ] for column_name in column_names
        }

        pd.DataFrame(log_dict).to_csv(
            output_directory / 'infer.csv', index=None)
        return


class FEMDataSaveFunction(IInfererSaveFunction):
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        output_directory: pathlib.Path,
        fem_data: femio.FEMData,
        overwrite: bool,
        write_simulation_type: str,
        *,
        less_output: bool = False
    ) -> None:
        ext = self.get_save_fem_extension(write_simulation_type)

        if less_output:
            self._reduce_fem_data(fem_data)

        fem_data.write(
            write_simulation_type,
            output_directory / ('mesh' + ext),
            overwrite=overwrite
        )
        return

    def get_save_fem_extension(self, write_simulation_type: str) -> str:
        # TODO: This function should be implemented in setting class
        if write_simulation_type == 'fistr':
            return ''
        elif write_simulation_type == 'ucd':
            return '.inp'
        elif write_simulation_type == 'vtk':
            return '.vtk'
        elif write_simulation_type in ['polyvtk', 'vtu']:
            return '.vtu'
        else:
            raise ValueError(
                f"Unexpected write_simulation_type: {write_simulation_type}")

    def _reduce_fem_data(self, fem_data: femio.FEMData) -> None:
        nodal_data = {}
        registered_keys = ['answer_', 'predicted_', 'difference_']
        for key in fem_data.nodal_data.keys():
            for registered_name in registered_keys:
                if registered_name not in key:
                    continue

                nodal_data.update(
                    {key: fem_data.nodal_data.get_attribute_data(key)}
                )

        fem_data.nodal_data.reset()
        elemental_data = {}
        for key in fem_data.elemental_data.keys():
            for registered_name in registered_keys:
                if registered_name not in key:
                    continue

                elemental_data.update({
                    key: fem_data.elemental_data.get_attribute_data(key)})

        has_face = ('face' in fem_data.elemental_data)
        if has_face:
            face = fem_data.elemental_data['face']

        fem_data.elemental_data.reset()
        fem_data.nodal_data.update_data(fem_data.nodes.ids, nodal_data)
        fem_data.elemental_data.update_data(
            fem_data.elements.ids, elemental_data
        )
        if has_face:
            fem_data.elemental_data['face'] = face

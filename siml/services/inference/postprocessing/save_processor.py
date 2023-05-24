import abc
import pathlib
from typing import Optional

import femio
import numpy as np
import pandas as pd
from ignite.engine import State

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


class WrapperResultItems:
    def __init__(
        self,
        inner_setting: InnerInfererSetting,
        result_state:  State
    ) -> None:
        self._state = result_state
        self._inner_setting = inner_setting

        self._output_dirs = []
        self._collect()

    def __len__(self):
        return len(self._state.metrics["post_results"])

    def _collect(self):
        for idx in range(len(self)):
            record = self.get_post_record(idx)
            each_output_directory = \
                self._inner_setting.get_output_directory(
                    record.inference_start_datetime,
                    data_directory=record.data_directory,
                )
            self._output_dirs.append(each_output_directory)

    def get_output_directory(self, idx: int) -> pathlib.Path:
        return self._output_dirs[idx]

    def get_post_record(self, idx: int) -> PostPredictionRecord:
        return self._state.metrics["post_results"][idx]

    def get_metrics(self, name: str) -> list[float]:
        return self._state.metrics[name]

    def get_item(self, idx: int, name: str):
        if name in PostPredictionRecord._fields:
            record = self.get_post_record(idx)
            return getattr(record, name)

        if name == "output_directory":
            return self._output_dirs[idx]

        if name in self._state.metrics:
            return self._state.metrics[name][idx]

        raise KeyError(f"{name}")


class SaveProcessor():
    def __init__(
        self,
        inner_setting: InnerInfererSetting,
        user_save_function: Optional[IInfererSaveFunction] = None
    ) -> None:
        self._inner_setting = inner_setting
        self._inferer_setting = inner_setting.inferer_setting
        self._conversion_setting = inner_setting.conversion_setting
        self._user_save_function = user_save_function

    def run(
        self,
        result_state: State,
    ) -> None:

        results = WrapperResultItems(
            self._inner_setting,
            result_state
        )
        # Save each results
        self.save_each_results(results)

        # Save overall settings
        output_directory = self._inner_setting.get_output_directory(
            date_string=results.get_item(0, "inference_start_datetime")
        )
        self._save_settings(output_directory)

        self._save_logs(
            output_directory=output_directory,
            results=results
        )

    def save_each_results(
        self,
        results: WrapperResultItems
    ):
        for idx in range(len(results)):
            record = results.get_post_record(idx)
            output_directory = results.get_output_directory(idx)

            self.save_npy_variables(
                dict_data_x=record.dict_x,
                dict_data_y=record.dict_y,
                output_directory=output_directory
            )

            fem_data = record.fem_data
            if fem_data is None:
                continue

            self.save_fem_data(
                output_directory=output_directory,
                fem_data=fem_data
            )

    def save_npy_variables(
        self,
        dict_data_x: dict,
        dict_data_y: dict,
        output_directory: pathlib.Path
    ) -> None:
        self._save_npy_data(dict_data_x, output_directory)
        self._save_npy_data(dict_data_y, output_directory)
        return

    def save_fem_data(
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
        results: WrapperResultItems,
        output_directory: pathlib.Path
    ) -> None:
        column_names = [
            'loss', 'raw_loss', 'output_directory', 'data_directory',
            'inference_time'
        ]

        log_dict = {}
        for column_name in column_names:
            log_dict.update(
                {
                    column_name: [
                        results.get_item(idx, column_name)
                        for idx in range(len(results))
                    ]
                }
            )

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

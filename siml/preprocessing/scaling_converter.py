from __future__ import annotations
import multiprocessing as multi
import pathlib
import pickle
from functools import partial
from typing import Optional

import pydantic
import pydantic.dataclasses as dc
from siml import setting, util
from siml.path_like_objects import SimlDirectory, ISimlNumpyFile
from siml.siml_variables import ArrayDataType
from siml.services.path_rules import SimlPathRules
from siml.base.siml_enums import DirectoryType

from .siml_scalers import IScalingSaveFunction, DefaultSaveFunction
from .scalers_composition import ScalersComposition


class Config:
    init = True
    frozen = True
    arbitrary_types_allowed = True


@dc.dataclass(config=Config)
class PreprocessInnerSettings():
    preprocess_dict: dict
    interim_directories: list[pathlib.Path]
    preprocessed_root: pathlib.Path
    recursive: bool = True
    REQUIRED_FILE_NAMES: Optional[list[str]] = None
    FINISHED_FILE: str = 'preprocessed'
    PREPROCESSORS_PKL_NAME: str = 'preprocessors.pkl'

    # cached value
    cached_interim_directories: Optional[list[SimlDirectory]] = None

    @pydantic.validator("REQUIRED_FILE_NAMES")
    def default_list_check(cls, v):
        if v is None:
            return ["converted"]
        return v

    @pydantic.root_validator()
    def _validate_interim_directories(cls, values):
        if values.get("cached_interim_directories"):
            ValueError(
                "Argument for cached_interim_directories is not allowed."
            )

        recursive = values["recursive"]
        interim_directories = values["interim_directories"]
        REQUIRED_FILE_NAMES = values['REQUIRED_FILE_NAMES']

        if recursive:
            siml_interim_directories = util.collect_data_directories(
                interim_directories,
                required_file_names=REQUIRED_FILE_NAMES
            )
        else:
            siml_interim_directories = interim_directories

        if len(siml_interim_directories) == 0:
            raise ValueError(
                'No converted data found. Perform conversion first.'
            )

        siml_directories = [
            SimlDirectory(p)
            for p in siml_interim_directories
        ]
        values["cached_interim_directories"] = siml_directories
        return values

    def get_default_preprocessors_pkl_path(self) -> pathlib.Path:
        preprocessor_pkl = self.preprocessed_root \
            / self.PREPROCESSORS_PKL_NAME
        return preprocessor_pkl

    def collect_interim_directories(self) -> list[SimlDirectory]:
        return self.cached_interim_directories

    def get_scaler_fitting_files(
        self,
        variable_name: str
    ) -> list[ISimlNumpyFile]:

        preprocess_setting = self.preprocess_dict[variable_name]
        siml_directories = self.collect_interim_directories()

        data_files = [
            siml_dir.find_variable_file(variable_name)
            for siml_dir in siml_directories
        ]
        for other_component in preprocess_setting['other_components']:
            data_files += [
                siml_dir.find_variable_file(other_component)
                for siml_dir in siml_directories
            ]
        return data_files

    def get_output_directory(
        self,
        data_directory: pathlib.Path
    ) -> pathlib.Path:
        rules = SimlPathRules()
        output_directory = rules.determine_output_directory(
            data_directory,
            self.preprocessed_root,
            allowed_type=DirectoryType.INTERIM
        )
        return output_directory


class ScalingConverter:
    """
    This is Facade Class for scaling process
    """
    @classmethod
    def read_settings(cls, settings_yaml: pathlib.Path, **args):
        main_setting = setting.MainSetting.read_settings_yaml(
            settings_yaml, replace_preprocessed=False)
        return cls(main_setting, **args)

    @classmethod
    def read_pkl(
        cls,
        main_setting: setting.MainSetting,
        converter_parameters_pkl: pathlib.Path,
        key: bytes = None,
    ):
        scalers = ScalersComposition.create_from_file(
            converter_parameters_pkl=converter_parameters_pkl,
            key=key
        )
        return cls(
            main_setting=main_setting,
            scalers=scalers
        )

    def __init__(
        self,
        main_setting: setting.MainSetting,
        *,
        force_renew: bool = False,
        save_func: Optional[IScalingSaveFunction] = None,
        max_process: int = None,
        allow_missing: bool = False,
        recursive: bool = True,
        scalers: Optional[ScalersComposition] = None
    ) -> None:
        """
        Initialize ScalingConverter

        Parameters
        ----------
            main_setting (setting.MainSetting): setting class
            force_renew: bool, optional
                If True, renew npy files even if they are alerady exist.
            recursive: bool, optional
                If True, search data recursively.
            save_func: callable, optional
                Callback function to customize save data. It should accept
                output_directory, variable_name, and transformed_data.
            max_process: int, optional
                The maximum number of processes.
            allow_missing: bool, optional
                If True, continue even if some of variables are missing.
        """

        self._setting = PreprocessInnerSettings(
            preprocess_dict=main_setting.preprocess,
            interim_directories=main_setting.data.interim,
            preprocessed_root=main_setting.data.preprocessed_root,
            recursive=recursive
        )

        if scalers is None:
            self._scalers: ScalersComposition \
                = ScalersComposition.create_from_dict(
                    preprocess_dict=main_setting.preprocess,
                    max_process=max_process,
                    key=main_setting.data.encrypt_key
                )
        else:
            self._scalers = scalers

        self._decrypt_key = main_setting.data.encrypt_key
        self.max_process = max_process
        self.force_renew = force_renew
        self.save_func = self._initialize_save_function(save_func)
        self.allow_missing = allow_missing

    def _initialize_save_function(
        self,
        function: Optional[IScalingSaveFunction]
    ) -> IScalingSaveFunction:
        if function is None:
            return DefaultSaveFunction()
        else:
            return function

    def fit_transform(
        self,
        group_id: Optional[int] = None
    ) -> None:
        """This function is consisted of these three process.
        - Determine parameters of scalers by reading data files lazily
        - Transform interim data and save result
        - Save file of parameters

        Parameters
        ----------
        group_id: int, optional
            group_id to specify chunk of preprocessing group. Useful when
            MemoryError occurs with all variables preprocessed in one node.
            If not specified, process all variables.

        Returns
        -------
        None
        """
        self.lazy_fit_all(group_id=group_id)
        self.transform_interim(group_id=group_id)
        self.save()

    def lazy_fit_all(
        self,
        *,
        group_id: int = None
    ) -> None:
        """Determine preprocessing parameters
        by reading data files lazily.

        Parameters
        ----------
        group_id: int, optional
            group_id to specify chunk of preprocessing group. Useful when
            MemoryError occurs with all variables preprocessed in one node.
            If not specified, process all variables.

        Returns
        -------
        None
        """

        scaler_name_to_files = {
            k: self._setting.get_scaler_fitting_files(k)
            for k in self._scalers.get_scaler_names(group_id=group_id)
        }
        self._scalers.lazy_partial_fit(scaler_name_to_files)

    def transform_interim(
        self,
        *,
        group_id: int = None
    ) -> None:
        """
        Apply scaling process to data in interim directory and save results
        in preprocessed directory.

        Parameters
        ----------
            group_id: int, optional
                group_id to specify chunk of preprocessing group. Useful when
                MemoryError occurs with all variables preprocessed in one node.
                If not specified, process all variables.

        Returns
        -------
        None
        """

        interim_dirs = self._setting.collect_interim_directories()
        variable_names = self._scalers.get_variable_names(group_id=group_id)

        # Parallel by scaling
        with multi.Pool(self.max_process) as pool:
            pool.map(
                partial(
                    self._transform_directories,
                    directories=interim_dirs
                ),
                variable_names,
                chunksize=1
            )

    def inverse_transform(
        self,
        dict_data: dict[str, ArrayDataType]
    ) -> dict[str, ArrayDataType]:
        return self._scalers.inverse_transform(dict_data)

    def save(self) -> None:
        """
        Save Parameters of scaling converters
        """
        dump_dict = self._scalers.get_dumped_object()
        pkl_path = self._setting.get_default_preprocessors_pkl_path()
        with open(pkl_path, 'wb') as f:
            pickle.dump(dump_dict, f)

    def _transform_directories(
        self,
        variable_name: str,
        directories: list[pathlib.Path]
    ) -> None:

        for siml_dir in directories:
            self._transform_single_directory(
                variable_name,
                siml_dir
            )

    def _transform_single_directory(
        self,
        variable_name: str,
        interim_dir: SimlDirectory
    ) -> None:
        siml_output_dir = SimlDirectory(
            self._setting.get_output_directory(interim_dir.path)
        )
        if self._can_skip(siml_output_dir, variable_name):
            return

        siml_file = interim_dir.find_variable_file(
            variable_name,
            allow_missing=self.allow_missing
        )
        if siml_file is None:
            return

        transformed_data = self._scalers.transform_file(
            variable_name,
            siml_file
        )

        self.save_func(
            output_directory=siml_output_dir.path,
            file_basename=variable_name,
            data=transformed_data,
            encrypt_key=self._decrypt_key
        )

    def _can_skip(
        self,
        output_dir: SimlDirectory,
        variable_name: str
    ) -> bool:
        if self.force_renew:
            return False

        if output_dir.exist_variable_file(variable_name):
            print(
                f"{output_dir.path} / {variable_name} "
                'already exists. Skipped.'
            )
            return True

        return False

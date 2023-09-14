from __future__ import annotations

import abc
import multiprocessing as multi
import pathlib
from functools import cache, partial, reduce
from operator import or_
from typing import Dict, Optional, Tuple, Union

import femio
import numpy as np

from siml import setting, util
from siml.base.siml_enums import DirectoryType
from siml.services.path_rules import SimlPathRules
from siml.utils import fem_data_utils


class IConvertFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(
        self,
        fem_data: femio.FEMData,
        data_directory: pathlib.Path
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError()


class ILoadFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(
        self,
        data_files: list[pathlib.Path],
        raw_path: pathlib.Path
    ) -> tuple[Dict, femio.FEMData]:
        raise NotImplementedError()


class ISaveFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(
        self,
        fem_data: femio.FEMData,
        dict_data: dict[str, np.ndarray],
        output_directory: pathlib.Path,
        force_renew: bool
    ) -> None:
        raise NotImplementedError()


class IFilterFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(
        self,
        fem_data: femio.FEMData,
        raw_path: pathlib.Path,
        dict_data: dict[str, np.ndarray]
    ) -> bool:
        raise NotImplementedError()


class SingleDataConverter:
    def __init__(
        self,
        setting: setting.ConversionSetting,
        raw_path: pathlib.Path,
        load_function: ILoadFunction,
        filter_function: IFilterFunction,
        *,
        save_function: ISaveFunction = None,
        output_directory: pathlib.Path = None,
        raise_when_overwrite: bool = False,
        force_renew: bool = False,
        return_results: bool = False
    ) -> None:

        self.setting = setting
        self.raw_path = raw_path
        self.load_function = load_function
        self.save_function = save_function

        self.filter_function = filter_function
        self._output_directory = output_directory
        self.force_renew = force_renew
        self.raise_when_overwrite = raise_when_overwrite
        self.return_results = return_results

        save_results = (not return_results)
        if save_results and (self.save_function is None):
            raise ValueError(
                "save_function is None when save_results option is True."
            )

    @property
    @cache
    def output_directory(self) -> pathlib.Path:
        if self._output_directory is not None:
            return self._output_directory
        else:
            rules = SimlPathRules()
            return rules.determine_output_directory(
                self.raw_path,
                self.setting.output_base_directory,
                allowed_type=DirectoryType.RAW
            )

    def run(self) -> Union[tuple[dict, femio.FEMData], None]:

        values = self._convert()
        if values is None:
            print(
                f"Conversion process for {self.raw_path} has failed"
            )
            return None

        dict_data, fem_data = values
        if self.return_results:
            return (dict_data, fem_data)

        self.save_function(
            fem_data,
            dict_data,
            self.output_directory,
            self.force_renew
        )

    def _convert(
        self
    ) -> Union[Tuple[dict[str, np.ndarray], femio.FEMData], None]:
        """
        conversion process. Convert to feature array from data files

        Raises:
            ValueError: When load function raise Exception

        Returns:
            Union[Tuple[dict[str, np.ndarray], femio.FEMData], None]:
                When success, return dict_data and fem_data.
                When failed, return empty dictionary and None
        """
        is_valid = self._check_directory()
        if not is_valid:
            return {}, None

        data_files = util.collect_files(
            self.raw_path,
            self.setting.required_file_names
        )
        try:
            dict_data, fem_data = self.load_function(
                data_files, self.raw_path
            )
        except BaseException as e:
            raise ValueError(
                f"{e}\nload_function failed for: {self.raw_path}"
            )

        if not self.filter_function(fem_data, self.raw_path, dict_data):
            return None

        if self.setting.should_load_mandatory_variables:
            wrapped_data = fem_data_utils.FemDataWrapper(fem_data)
            mandatory_dict = wrapped_data.extract_variables(
                self.setting.mandatory_variables,
                optional_variables=self.setting.optional_variables
            )
            dict_data.update(mandatory_dict)
        return dict_data, fem_data

    def _check_directory(self) -> bool:

        # check raw path
        valid_raw_path = self._check_raw_path()
        if not valid_raw_path:
            return False

        if self.return_results:
            return True

        # check output directory
        valid_output_directory = self._check_output_direcotry()
        return valid_output_directory

    def _check_raw_path(self) -> bool:
        # Check raw_path
        if self.raw_path.is_file():
            return True

        if self.raw_path.is_dir():
            if not util.files_exist(
                    self.raw_path,
                    self.setting.required_file_names
            ):
                print(
                    "No required files is found in raw_path: "
                    f"{self.raw_path}"
                )
                return False
            else:
                return True

        raise ValueError(f"raw_path not understandable: {self.raw_path}")

    def _check_output_direcotry(self):
        # check output directory
        finished_file = self.output_directory / self.setting.finished_file
        if not finished_file.exists():
            return True

        if self.force_renew:
            return True

        if self.raise_when_overwrite:
            raise ValueError(f"{self.output_directory} already exists.")

        print(f"Already converted. Skipped conversion: {self.raw_path}")
        return False


class RawConverter:

    @classmethod
    def read_settings(cls, settings_yaml, **args):
        main_setting = setting.MainSetting.read_settings_yaml(
            settings_yaml, replace_preprocessed=False)
        return cls(main_setting, **args)

    def __init__(
        self,
        main_setting: setting.MainSetting,
        *,
        recursive: bool = True,
        conversion_function: IConvertFunction = None,
        filter_function: IFilterFunction = None,
        load_function: ILoadFunction = None,
        save_function: ISaveFunction = None,
        force_renew: bool = False,
        read_npy: bool = False,
        write_ucd: bool = True,
        read_res: bool = True,
        max_process: int = None,
        to_first_order: bool = False
    ) -> None:
        """Initialize converter of raw data and save them in interim directory.

        Parameters
        ----------
        main_setting: siml.setting.MainSetting
            MainSetting object.
        recursive: bool, optional
            If True, recursively convert data.
        conversion_function: callable, optional
            Conversion function which takes femio.FEMData object and
            pathlib.Path (data directory) as only arguments and returns data
            dict to be saved.
        filter_function: callable, optional
            Function to filter the data which can be converted. It should take
            femio.FEMData object, pathlib.Path (data directory), and dict_data
            as only arguments and returns True (for convertable data) or False
            (for unconvertable data).
        load_function: callable, optional
            Function to load data, which take list of pathlib.Path objects
            (as required files) and pathlib.Path object (as data directory)
            and returns data_dictionary and fem_data (can be None) to be saved.
        save_function: callable, optional
            Additional function to save data, which take femio.FEMData object,
            data_dict, pathliub.Path object as output directory,
            and bool represents force renew.
            If fed, this function is run prior to default save function
        force_renew: bool, optional
            If True, renew npy files even if they are alerady exist.
        read_npy: bool, optional
            If True, read .npy files instead of original files if exists.
        write_ucd: bool, optional
            If True, write AVS UCD file with preprocessed variables.
        read_res: bool, optional
            If True, read res file of FrontISTR.
        max_process: int, optional
            The maximum number of processes to perform conversion.
        """
        self.main_setting: setting.MainSetting = main_setting
        self.setting = self.main_setting.conversion
        self.recursive = recursive
        self.conversion_function = conversion_function
        self.filter_function = filter_function
        self.load_function = load_function
        self.save_function = save_function
        self.force_renew = force_renew
        self.read_npy = read_npy
        self.write_ucd = write_ucd
        self.to_first_order = to_first_order
        self.read_res = read_res
        self.max_process = min(
            main_setting.conversion.max_process,
            util.determine_max_process(max_process))
        self.setting.output_base_directory \
            = main_setting.data.interim_root

        self._check_args()

    def _check_args(self) -> None:

        if (self.load_function is not None) and \
                (self.conversion_function is not None):
            raise Exception(
                "conversion function and load_function "
                "cannot set at the same time"
            )

    def convert(
        self,
        raw_directory: pathlib.Path = None,
        *,
        return_results: bool = False
    ) -> dict[str, Union[tuple[dict, femio.FEMData], None]]:
        """Perform conversion.

        Parameters
        ----------
        raw_directory: pathlib.Path, optional
            Raw data directory name. If not fed, self.setting.data.raw is used
            instead.

        return_results: bool, optional
            If True, save results and dump files

        Returns
        -------
        dict[str, Union[dict, None]]:
            key is a path to raw directory.
            If return_results is False, values is a list of None.
            If return_results is True, values is a dictionary
             of converted values.
        """
        print(f"# process: {self.max_process}")
        raw_directories = self._search_raw_directories(raw_directory)
        chunksize = max(len(raw_directories) // self.max_process // 16, 1)

        with multi.Pool(self.max_process) as pool:
            results = pool.map(
                partial(
                    self.convert_single_data,
                    return_results=return_results
                ),
                raw_directories,
                chunksize=chunksize
            )

        flatten_results = reduce(lambda x, y: x | y, results)
        return flatten_results

    def convert_single_data(
        self,
        raw_path: pathlib.Path,
        *,
        output_directory: pathlib.Path = None,
        raise_when_overwrite: bool = False,
        return_results: bool = False
    ) -> dict[str, Union[tuple[dict, femio.FEMData], None]]:
        """Convert single directory.

        Parameters
        ----------
        raw_path: pathlib.Path
            Input data path of raw data.
        output_directory: pathlib.Path, optional
            If fed, use the fed path as the output directory.
        raise_when_overwrite: bool, optional
            If True, raise when the output directory exists. The default is
            False.

        Returns
        -------
        dict[str, Union[dict, None]]:
            key is a path to raw directory.
            If return_results is False, values is a list of None.
            If return_results is True, values is a dictionary
             of converted values.
        """
        load_function = self._create_load_function()
        save_function = self._create_save_function()
        filter_function = self._create_filter_function()

        single_converter = SingleDataConverter(
            setting=self.setting,
            raw_path=raw_path,
            load_function=load_function,
            save_function=save_function,
            filter_function=filter_function,
            output_directory=output_directory,
            raise_when_overwrite=raise_when_overwrite,
            force_renew=self.force_renew,
            return_results=return_results
        )

        result = single_converter.run()
        return {str(raw_path): result}

    def _search_raw_directories(
        self,
        raw_directory: pathlib.Path = None
    ) -> list[pathlib.Path]:

        if raw_directory is None:
            raw_directory = self.main_setting.data.raw

        # Process all subdirectories when recursice is True
        if self.recursive:
            if isinstance(raw_directory, (list, tuple, set)):
                raw_directories = reduce(
                    or_,
                    [
                        set(
                            util.collect_data_directories(
                                pathlib.Path(d),
                                print_state=True
                            )
                        )
                        for d in raw_directory
                    ]
                )
            else:
                raw_directories = util.collect_data_directories(
                    pathlib.Path(raw_directory), print_state=True)
        else:
            if isinstance(raw_directory, (list, tuple, set)):
                raw_directories = raw_directory
            else:
                raw_directories = [raw_directory]

        return raw_directories

    def _create_load_function(self) -> ILoadFunction:

        if self.load_function is not None:
            return self.load_function

        load_function = DefaultLoadFunction(
            file_type=self.setting.file_type,
            read_npy=self.read_npy,
            read_res=self.read_res,
            skip_femio=self.setting.skip_femio,
            time_series=self.setting.time_series,
            conversion_function=self.conversion_function
        )

        return load_function

    def _create_save_function(self) -> DefaultSaveFunction:
        default_save_function = DefaultSaveFunction(
            main_setting=self.main_setting,
            write_ucd=self.write_ucd,
            to_first_order=self.to_first_order,
            user_save_function=self.save_function
        )
        return default_save_function

    def _create_filter_function(self) -> IFilterFunction:
        if self.filter_function is not None:
            return self.filter_function

        default_filter_function = DefaultFilterFunction()
        return default_filter_function


class DefaultLoadFunction(ILoadFunction):
    def __init__(
        self,
        file_type: str,
        read_npy: bool,
        read_res: bool,
        skip_femio: bool,
        time_series: bool,
        conversion_function: IConvertFunction = None
    ) -> None:
        self.file_type = file_type
        self.read_npy = read_npy
        self.read_res = read_res
        self.skip_femio = skip_femio
        self.time_series = time_series
        self.conversion_function = conversion_function

    def __call__(
        self,
        data_files: list[pathlib.Path],
        raw_path: pathlib.Path
    ) -> tuple[Dict, femio.FEMData]:

        fem_data = self._prepare_fem_data(raw_path)

        if self.conversion_function is None:
            return {}, fem_data

        try:
            dict_data = self.conversion_function(fem_data, raw_path)
        except BaseException as e:
            raise ValueError(
                f"{e}\nconversion_function failed for: {raw_path}")

        return dict_data, fem_data

    def _prepare_fem_data(
        self,
        raw_path: pathlib.Path
    ) -> Union[None, femio.FEMData]:
        if self.skip_femio:
            return None

        try:
            fem_data = self._load_fem_data(raw_path)
        except ValueError:
            print(f"femio read failed. Skipped.: {raw_path}")
            fem_data = None
        except BaseException as ex:
            print(f"femio read failed. : {raw_path}")
            raise ex
        return fem_data

    def _load_fem_data(
        self,
        raw_path: pathlib.Path
    ) -> femio.FEMData:

        if raw_path.is_dir():
            fem_data = femio.FEMData.read_directory(
                self.file_type,
                raw_path,
                read_npy=self.read_npy,
                save=False,
                read_res=self.read_res,
                time_series=self.time_series)
        else:
            fem_data = femio.FEMData.read_files(
                self.file_type,
                raw_path,
                time_series=self.time_series)

        return fem_data


class DefaultSaveFunction(ISaveFunction):
    def __init__(
        self,
        main_setting: setting.MainSetting,
        write_ucd: bool,
        to_first_order: bool,
        *,
        user_save_function: Optional[ISaveFunction] = None
    ) -> None:
        self.main_setting = main_setting
        self.setting = main_setting.conversion
        self.write_ucd = write_ucd
        self.to_first_order = to_first_order
        self.user_save_function = user_save_function

    def __call__(
        self,
        fem_data: femio.FEMData,
        dict_data: dict[str, np.ndarray],
        output_directory: pathlib.Path,
        force_renew: bool
    ) -> None:
        if fem_data is not None:
            self._save_fem_data(
                fem_data,
                dict_data,
                output_directory=output_directory,
                force_renew=force_renew
            )

        if self.user_save_function is not None:
            self.user_save_function(
                fem_data,
                dict_data,
                output_directory=output_directory,
                force_renew=force_renew
            )

        if self.setting.skip_save:
            return

        if len(dict_data) == 0:
            output_directory.mkdir(parents=True, exist_ok=True)
            (output_directory / "failed").touch()
            return

        save_dict_data(
            output_directory=output_directory,
            dict_data=dict_data,
            encrypt_key=self.main_setting.data.encrypt_key,
            finished_file=self.setting.finished_file,
            save_dtype_dict=self.main_setting.misc.get("save_dtype_dict")
        )
        return

    def _save_fem_data(
        self,
        fem_data: femio.FEMData,
        dict_data: dict,
        output_directory: pathlib.Path,
        force_renew: bool
    ) -> None:
        if self.setting.save_femio:
            fem_data.save(output_directory)

        if self.write_ucd:
            if self.to_first_order:
                fem_data_to_save = fem_data.to_first_order()
            else:
                fem_data_to_save = fem_data

            wrapped_data = fem_data_utils.FemDataWrapper(
                fem_data_to_save
            )
            wrapped_data.update_fem_data(
                dict_data, allow_overwrite=True
            )
            updated_fem_data_to_save = wrapped_data.fem_data
            updated_fem_data_to_save.to_first_order().write(
                'ucd',
                output_directory / 'mesh.inp',
                overwrite=force_renew
            )


class DefaultFilterFunction(IFilterFunction):
    def __init__(self) -> None:
        pass

    def __call__(
            self,
            fem_data: femio.FEMData,
            raw_path: pathlib.Path,
            dict_data: dict[str, np.ndarray]
    ) -> bool:
        return True


def save_dict_data(
        output_directory: pathlib.Path,
        dict_data: dict[str, np.ndarray],
        *,
        dtype=np.float32,
        encrypt_key=None,
        finished_file='converted',
        save_dtype_dict: Dict = None
) -> None:
    """Save dict_data.

    Parameters
    ----------
    output_directory: pathlib.Path
        Output directory path.
    dict_data: dict
        Data dictionary to be saved.
    dtype: type, optional
        Data type to be saved.
    encrypt_key: bytes, optional
        Data for encryption.

    Returns
    -------
        None
    """
    for key, value in dict_data.items():
        save_dtype = _get_save_dtype(
            key,
            default_dtype=dtype,
            save_dtype_dict=save_dtype_dict
        )
        util.save_variable(
            output_directory,
            key,
            value,
            dtype=save_dtype,
            encrypt_key=encrypt_key
        )
    (output_directory / finished_file).touch()
    return


def _get_save_dtype(
    variable_name: str,
    default_dtype: np.dtype,
    save_dtype_dict: Dict = None
) -> np.dtype:
    if save_dtype_dict is None:
        return default_dtype
    if variable_name in save_dtype_dict:
        return save_dtype_dict[variable_name]
    else:
        return default_dtype

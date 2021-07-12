"""Module for preprocessing."""

import datetime as dt
from functools import reduce
import glob
import io
import itertools as it
import multiprocessing as multi
from operator import or_
import os
from pathlib import Path
import pickle

import femio
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from . import util
from . import setting


FEMIO_FILE = 'femio_npy_saved.npy'


class RawConverter():

    @classmethod
    def read_settings(cls, settings_yaml, **args):
        main_setting = setting.MainSetting.read_settings_yaml(
            settings_yaml, replace_preprocessed=False)
        return cls(main_setting, **args)

    def __init__(
            self, main_setting, *,
            recursive=True,
            conversion_function=None, filter_function=None, load_function=None,
            save_function=None,
            force_renew=False, read_npy=False, write_ucd=True, read_res=True,
            max_process=None, to_first_order=False):
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
            Function to save data, which take femio.FEMData object,
            data_dict, pathliub.Path object as output directory,
            and bool represents force renew.
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
        self.setting = main_setting
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
        self.max_process = util.determine_max_process(max_process)
        self.setting.conversion.output_base_directory \
            = self.setting.data.interim_root

    def convert(self, raw_directory=None):
        """Perform conversion.

        Parameters
        ----------
        raw_directory: str or pathlib.Path, optional
            Raw data directory name. If not fed, self.setting.data.raw is used
            instead.
        """
        if raw_directory is None:
            raw_directory = self.setting.data.raw

        print(raw_directory)

        # Process all subdirectories when recursice is True
        if self.recursive:
            if isinstance(raw_directory, (list, tuple, set)):
                raw_directories = reduce(or_, [
                    set(util.collect_data_directories(Path(d)))
                    for d in raw_directory])
            else:
                raw_directories = util.collect_data_directories(
                    Path(raw_directory))
        else:
            if isinstance(raw_directory, (list, tuple, set)):
                raw_directories = raw_directory
            else:
                raw_directories = [raw_directory]

        chunksize = max(len(raw_directories) // self.max_process // 16, 1)

        with multi.Pool(self.max_process) as pool:
            pool.map(
                self.convert_single_data, raw_directories,
                chunksize=chunksize)

        return

    def convert_single_data(
            self, raw_path, *, output_directory=None,
            raise_when_overwrite=False):
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
        None
        """
        conversion_setting = self.setting.conversion

        # Determine output directory
        raw_path = Path(raw_path)
        print(f"Processing: {raw_path}")
        if output_directory is None:
            output_directory = determine_output_directory(
                raw_path, conversion_setting.output_base_directory, 'raw')

        # Guard
        if raw_path.is_dir():
            if not util.files_exist(
                    raw_path, conversion_setting.required_file_names):
                return
        elif raw_path.is_file():
            pass
        else:
            raise ValueError(f"raw_path not understandable: {raw_path}")

        if (output_directory / conversion_setting.finished_file).exists():
            if raise_when_overwrite:
                raise ValueError(f"{output_directory} already exists.")
            if not self.force_renew:
                print(
                    f"Already converted. Skipped conversion: {raw_path}")
                return

        # Main process
        if conversion_setting.skip_femio:
            fem_data = None
            dict_data = {}
        else:
            if self.read_npy and (output_directory / FEMIO_FILE).exists():
                fem_data = femio.FEMData.read_npy_directory(
                    output_directory)
            else:
                try:
                    if raw_path.is_dir():
                        fem_data = femio.FEMData.read_directory(
                            conversion_setting.file_type, raw_path,
                            read_npy=self.read_npy, save=False,
                            read_res=self.read_res,
                            time_series=conversion_setting.time_series)
                    else:
                        fem_data = femio.FEMData.read_files(
                            conversion_setting.file_type, raw_path,
                            time_series=conversion_setting.time_series)
                except ValueError:
                    print("femio read failed. Skipped.")
                    output_directory.mkdir(parents=True, exist_ok=True)
                    (output_directory / 'failed').touch()
                    return

            if conversion_setting.mandatory_variables is not None \
                    and len(conversion_setting.mandatory_variables) > 0:
                dict_data = extract_variables(
                    fem_data, conversion_setting.mandatory_variables,
                    optional_variables=conversion_setting.optional_variables
                )
            else:
                dict_data = {}

        if self.conversion_function is not None:
            dict_data.update(
                self.conversion_function(fem_data, raw_path))

        if self.load_function is not None:
            data_files = util.collect_files(
                raw_path, conversion_setting.required_file_names)
            loaded_dict_data, fem_data = self.load_function(
                data_files, raw_path)
            dict_data.update(loaded_dict_data)

        if self.filter_function is not None and not self.filter_function(
                fem_data, raw_path, dict_data):
            return

        # Save data
        output_directory.mkdir(parents=True, exist_ok=True)
        if fem_data is not None:
            if self.setting.conversion.save_femio:
                fem_data.save(output_directory)

            if self.write_ucd:
                if self.to_first_order:
                    fem_data_to_save = fem_data.to_first_order()
                else:
                    fem_data_to_save = fem_data
                fem_data_to_save = update_fem_data(
                    fem_data_to_save, dict_data, allow_overwrite=True)
                fem_data_to_save.to_first_order().write(
                    'ucd', output_directory / 'mesh.inp',
                    overwrite=self.force_renew)
        if self.save_function is not None:
            self.save_function(
                fem_data, dict_data, output_directory, self.force_renew)

        if not self.setting.conversion.skip_save:
            save_dict_data(
                output_directory, dict_data,
                encrypt_key=self.setting.data.encrypt_key,
                finished_file=self.setting.conversion.finished_file)

        return


def update_fem_data(fem_data, dict_data, prefix='', *, allow_overwrite=False):
    for key, value in dict_data.items():

        variable_name = prefix + key
        if isinstance(value, np.ndarray):
            if len(value.shape) > 2 and value.shape[-1] == 1:
                if len(value.shape) == 4 and value.shape[1] == 3 \
                        and value.shape[2] == 3:
                    # NOTE: Assume this is symmetric matrix
                    reshaped_value \
                        = fem_data.convert_symmetric_matrix2array(
                            value[..., 0])
                else:
                    reshaped_value = value[..., 0]
                dict_data_to_update = {
                    variable_name: value,
                    variable_name + '_reshaped': reshaped_value}
            else:
                dict_data_to_update = {
                    variable_name: value}
            len_data = len(value)

            if len_data == len(fem_data.nodes.ids):
                # Nodal data
                fem_data.nodal_data.update_data(
                    fem_data.nodes.ids, dict_data_to_update,
                    allow_overwrite=allow_overwrite)
            elif len_data == len(fem_data.elements.ids):
                # Elemental data
                fem_data.elemental_data.update_data(
                    fem_data.elements.ids, dict_data_to_update,
                    allow_overwrite=allow_overwrite)
            else:
                print(f"{variable_name} is skipped to include in fem_data")
                continue
        else:
            print(f"{variable_name} is skipped to include in fem_data")

    return fem_data


def add_difference(
        fem_data, dict_data, reference_dict_data, prefix='difference'):
    intersections = set(
        dict_data.keys()).intersection(reference_dict_data.keys())
    if len(intersections) == 0:
        return fem_data

    difference_dict_data = {
        intersection:
        np.reshape(
            dict_data[intersection], reference_dict_data[intersection].shape)
        - reference_dict_data[intersection]
        for intersection in intersections}
    fem_data = update_fem_data(fem_data, difference_dict_data, prefix=prefix)

    return fem_data


def concatenate_preprocessed_data(
        preprocessed_base_directories, output_directory_base, variable_names,
        *, ratios=(.9, .05, .05), overwrite=False):
    """Concatenate preprocessed data in the element direction.

    NOTE: It may lead data leakage so it is just for research use.

    Parameters
    ----------
    preprocessed_base_directories: pathlib.Path or list[pathlib.Path]
        Base directory name of preprocessed data.
    output_directory_base: pathlib.Path
        Base directory of output. Inside of it, train, validation, and
        test directories will be created.
    variable_names: list[str]
        Variable names to be concatenated.
    ratios: list[float], optional
        Ratio to split data.
    overwrite: bool, optional
        If True, overwrite output data.
    """
    if np.abs(np.sum(ratios) - 1.0) > 1e-5:
        raise ValueError('The sum of ratios does not make 1.')
    preprocessed_directories = util.collect_data_directories(
        preprocessed_base_directories,
        required_file_names=Preprocessor.FINISHED_FILE)
    dict_data = {
        variable_name:
        np.concatenate([
            util.load_variable(preprocessed_directory, variable_name)
            for preprocessed_directory in preprocessed_directories])
        for variable_name in variable_names}

    data_length = len(dict_data[variable_names[0]])
    indices = np.arange(data_length)
    np.random.shuffle(indices)

    train_length = int(np.round(data_length * ratios[0]))
    validation_length = int(np.round(data_length * ratios[1]))
    test_length = data_length - train_length - validation_length

    (output_directory_base / 'train').mkdir(
        parents=True, exist_ok=True)
    (output_directory_base / 'validation').mkdir(
        parents=True, exist_ok=True)
    (output_directory_base / 'test').mkdir(
        parents=True, exist_ok=True)

    for variable_name, data in dict_data.items():
        np.save(
            output_directory_base / f"train/{variable_name}.npy",
            data[indices[:train_length]])
        if validation_length > 0:
            np.save(
                output_directory_base / f"validation/{variable_name}.npy",
                data[indices[train_length:train_length+validation_length]])
        if test_length > 0:
            np.save(
                output_directory_base / f"validation/{variable_name}.npy",
                data[indices[train_length+validation_length:]])
    return


class Preprocessor:

    REQUIRED_FILE_NAMES = ['converted']
    FINISHED_FILE = 'preprocessed'
    PREPROCESSORS_PKL_NAME = 'preprocessors.pkl'

    @classmethod
    def read_settings(cls, settings_yaml, **args):
        main_setting = setting.MainSetting.read_settings_yaml(
            settings_yaml, replace_preprocessed=False)
        return cls(main_setting, **args)

    def __init__(
            self, main_setting, force_renew=False, save_func=None,
            str_replace='interim', max_process=None, allow_missing=False):
        """Initialize preprocessor of interim data with preprocessing
        e.g. standardization and then save them.

        Parameters
        ----------
        force_renew: bool, optional
            If True, renew npy files even if they are alerady exist.
        save_func: callable, optional
            Callback function to customize save data. It should accept
            output_directory, variable_name, and transformed_data.
        str_replace: str, optional
            String to replace data directory in order to convert from interim
            data to preprocessed data.
        max_process: int, optional
            The maximum number of processes.
        allow_missing: bool, optional
            If True, continue even if some of variables are missing.
        """
        self.setting = main_setting
        self.force_renew = force_renew
        self.save_func = save_func
        self.interim_directories = util.collect_data_directories(
            self.setting.data.interim,
            required_file_names=self.REQUIRED_FILE_NAMES)
        self.str_replace = str_replace
        self.max_process = util.determine_max_process(max_process)
        self.allow_missing = allow_missing
        if len(self.interim_directories) == 0:
            raise ValueError(
                'No converted data found. Perform conversion first.')
        return

    def preprocess_interim_data(self):
        """Preprocess interim data.

        Parameters
        ----------
        None

        Returns
        -------
        dict_preprocessor_setting: dict
            dict describing settings and parameters for preprocessors.
        """
        self.prepare_preprocess_converters()
        dict_preprocessor_settings = \
            self.merge_dict_preprocessor_setting_pkls()
        self.convert_interim_data()
        return dict_preprocessor_settings

    def prepare_preprocess_converters(self, *, group_id=None):
        """Prepare preprocess converters by reading data files lazily to
        determine preprocessing parameters (like std and mean for
        StandardScaler, min and max for MinMaxScaler.

        Parameters
        ----------
        group_id: int, optional
            group_id to specify chunk of preprocessing group. Useful when
            MemoryError occurs with all variables preprocessed in one node.
            If not specified, process all variables.

        Returns
        -------
        dict_preprocessor_setting: dict
            dict describing settings and parameters for preprocessors.
        """
        preprocessor_inputs = [
            (variable_name, preprocess_setting)
            for variable_name, preprocess_setting
            in self.setting.preprocess.items()
            if group_id is None or preprocess_setting['group_id'] == group_id]

        with multi.Pool(self.max_process) as pool:
            list_dict_preprocessor_setting = pool.starmap(
                self.prepare_preprocess_converter, preprocessor_inputs,
                chunksize=1)

        dict_preprocessor_settings = {}
        for dict_preprocessor_setting in list_dict_preprocessor_setting:
            if dict_preprocessor_setting is not None:
                dict_preprocessor_settings.update(dict_preprocessor_setting)
        return dict_preprocessor_settings

    def merge_dict_preprocessor_setting_pkls(self, data_directory=None):
        """Merge variable-wise preprocessor settings pkl files into one file.

        Parameters
        ----------
        data_directory: pathlib.Path, optional
            Directory path contains variable-wise preprocessor settings pkl
            files. If not fed, looking at self.setting.data.preprocessed_root .

        Returns
        -------
        dict_preprocessor_setting: dict
            dict describing settings and parameters for preprocessors after
            merger.
        """
        if data_directory is None:
            data_directory = self.setting.data.preprocessed_root
        preprocessors_pkl_path = data_directory / self.PREPROCESSORS_PKL_NAME

        if self.force_renew or not preprocessors_pkl_path.is_file():
            pkl_files = glob.glob(
                str(data_directory / f"*_{self.PREPROCESSORS_PKL_NAME}"))

            dict_before_replacement = {}
            for pkl_file in pkl_files:
                with open(pkl_file, 'rb') as f:
                    dict_before_replacement.update(pickle.load(f))

            dict_preprocessor_settings = {
                variable_name: self._collect_reference_dict_setting(
                    variable_name, dict_before_replacement)
                for variable_name in dict_before_replacement.keys()}

            self.dump_preprocessors(
                dict_preprocessor_settings, preprocessors_pkl_path)

        else:
            print(f"{preprocessors_pkl_path} already exists. Skip merger.")
            with open(preprocessors_pkl_path, 'rb') as f:
                dict_preprocessor_settings = pickle.load(f)

        return dict_preprocessor_settings

    def convert_interim_data(
            self, preprocessor_pkl=None, *, group_id=None):
        """Convert interim data with the determined preprocessor_settings.

        Parameters
        ----------
        preprocessor_pkl: dict or pathlib.Path, optional
            dict or pickle file path describing settings and parameters for
            preprocessors. If not fed, data will be loaded from
            self.setting.data.preprocessed_root.
        group_id: int, optional
            group_id to specify chunk of preprocessing group. Useful when
            MemoryError occurs with all variables preprocessed in one node.
            If not specified, process all variables.

        Returns
        -------
        None
        """

        if preprocessor_pkl is None:
            preprocessor_pkl = self.setting.data.preprocessed_root \
                / self.PREPROCESSORS_PKL_NAME
            if not preprocessor_pkl.is_file():
                raise ValueError(f"{preprocessor_pkl} not found.")

        if isinstance(preprocessor_pkl, Path):
            with open(preprocessor_pkl, 'rb') as f:
                dict_preprocessor_settings = pickle.load(f)
        else:
            dict_preprocessor_settings = preprocessor_pkl

        preprocess_converter_inputs = \
            self._generate_preprocess_converter_inputs(
                dict_preprocessor_settings, group_id)
        with multi.Pool(self.max_process) as pool:
            pool.starmap(
                self.transform_single_variable, preprocess_converter_inputs,
                chunksize=1)

        # Touch finished files
        for data_directory in self.interim_directories:
            output_directory = determine_output_directory(
                data_directory, self.setting.data.preprocessed_root,
                self.str_replace)
            (output_directory / self.FINISHED_FILE).touch()

        yaml_file = self.setting.data.preprocessed_root / 'settings.yml'
        if not yaml_file.exists():
            setting.write_yaml(self.setting, yaml_file)

        return

    def _generate_preprocess_converter_inputs(
            self, dict_preprocessor_settings, group_id):

        preprocess_converter_inputs = [
            (
                variable_name,
                self._collect_preprocess_converter_input(
                    variable_name, dict_preprocessor_settings))
            for variable_name, setting in self.setting.preprocess.items()
            if group_id is None or setting['group_id'] == group_id]
        return preprocess_converter_inputs

    def _collect_preprocess_converter_input(
            self, variable_name, dict_preprocessor_settings):

        reference_dict = self._collect_reference_dict_setting(
            variable_name, dict_preprocessor_settings)

        preprocess_converter = util.PreprocessConverter(
            reference_dict['preprocess_converter'],
            method=reference_dict['method'],
            componentwise=reference_dict['componentwise'],
            power=reference_dict.get('power', 1.),
            other_components=reference_dict['other_components'])
        if preprocess_converter is None:
            raise ValueError(f"Reference of {variable_name} is None")

        return preprocess_converter

    def _collect_reference_dict_setting(
            self, variable_name, dict_preprocessor_settings):
        if dict_preprocessor_settings[variable_name]['preprocess_converter'] \
                is None:
            value = dict_preprocessor_settings[variable_name]
            reference_name = self.setting.preprocess[variable_name]['same_as']
            if reference_name is None:
                raise ValueError(
                    f"Invalid setting for {variable_name}: {value}")
            reference_dict = dict_preprocessor_settings[reference_name]

        else:
            reference_dict = dict_preprocessor_settings[
                variable_name]

        return reference_dict

    def prepare_preprocess_converter(self, variable_name, preprocess_setting):
        """Prepare preprocess converter for single variable.

        Parameters
        ----------
        variable_name: str
            The name of the variable.
        preprocess_setting: dict
            Dictionary of preprocess setting contains 'method' and
            'componentwise' keywords.

        Returns
        -------
        dict_preprocessor_setting: dict
            Dict of preprocessor setting for the variable.
        """
        print(variable_name, preprocess_setting)

        # Check if data already exists
        pkl_file = self.setting.data.preprocessed_root \
            / self.PREPROCESSORS_PKL_NAME
        variable_pkl_file = self.setting.data.preprocessed_root \
            / f"{variable_name}_{self.PREPROCESSORS_PKL_NAME}"
        if not self.force_renew and (
                pkl_file.exists() or variable_pkl_file.exists()):
            print(
                'Data already exists in '
                f"{self.setting.data.preprocessed_root} for {variable_name}. "
                'Skipped.')
            return

        # Prepare preprocessor
        if (self.interim_directories[0] / (variable_name + '.npy')).exists():
            ext = '.npy'
        elif (
                self.interim_directories[0]
                / (variable_name + '.npy.enc')).exists():
            ext = '.npy.enc'
        elif (self.interim_directories[0] / (variable_name + '.npz')).exists():
            ext = '.npz'
        elif (
                self.interim_directories[0]
                / (variable_name + '.npz.enc')).exists():
            ext = '.npz.enc'
        else:
            raise ValueError(
                f"Unknown extension or file not found for {variable_name}")

        if preprocess_setting['same_as'] is None:
            if preprocess_setting['method'] == 'identity':
                preprocess_converter = util.PreprocessConverter(
                    'identity',
                    componentwise=preprocess_setting['componentwise'],
                    other_components=[],
                    power=1., key=self.setting.data.encrypt_key)
            else:
                data_files = [
                    data_directory / (variable_name + ext)
                    for data_directory in self.interim_directories]
                for other_component in preprocess_setting['other_components']:
                    data_files += [
                        data_directory / (other_component + ext)
                        for data_directory in self.interim_directories]
                preprocess_converter = util.PreprocessConverter(
                    preprocess_setting['method'], data_files=data_files,
                    componentwise=preprocess_setting['componentwise'],
                    power=preprocess_setting['power'],
                    other_components=preprocess_setting['other_components'],
                    key=self.setting.data.encrypt_key)
        else:
            # same_as is set so no need to prepare preprocessor
            preprocess_converter = None

        dict_preprocessor_setting = {
            variable_name: {
                'method': preprocess_setting['method'],
                'componentwise': preprocess_setting['componentwise'],
                'preprocess_converter': preprocess_converter,
                'power': preprocess_setting['power'],
                'other_components': preprocess_setting['other_components'],
            }}
        if not self.setting.data.preprocessed_root.exists():
            self.setting.data.preprocessed_root.mkdir(
                parents=True, exist_ok=True)
        partial_pkl_name = self.setting.data.preprocessed_root \
            / f"{variable_name}_{self.PREPROCESSORS_PKL_NAME}"
        self.dump_preprocessors(dict_preprocessor_setting, partial_pkl_name)

        return dict_preprocessor_setting

    def dump_preprocessors(self, dict_preprocessor_setting, file_path):
        dict_to_dump = {}
        for key, value in dict_preprocessor_setting.items():
            dict_to_dump[key] = {}
            for k, v in value.items():
                if k == 'preprocess_converter' and v is not None:
                    if isinstance(v, dict):
                        dict_to_dump[key].update({k: v})
                    else:
                        dict_to_dump[key].update({k: vars(v.converter)})
                else:
                    dict_to_dump[key].update({k: v})
        with open(file_path, 'wb') as f:
            pickle.dump(dict_to_dump, f)

        return

    def _file_exists(self, output_directory, variable_name):
        npy_file = output_directory / (variable_name + '.npy')
        npy_enc_file = output_directory / (
            variable_name + '.npy.enc')
        npz_file = output_directory / (variable_name + '.npz')
        npz_enc_file = output_directory / (
            variable_name + '.npz.enc')
        if npy_file.is_file():
            return True
        if npy_enc_file.is_file():
            return True
        if npz_file.is_file():
            return True
        if npz_enc_file.is_file():
            return True
        return False

    def transform_single_variable(self, variable_name, preprocess_converter):
        """Transform single variable with the created preprocess_converter.

        Parameters
        ----------
        variable_name: str
            The name of the variable.
        preprocess_converter: siml.util.PreprocessConverter
            The PreprocessConverter object to transform.

        Returns
        -------
        None
        """
        if isinstance(preprocess_converter.converter, util.Identity):
            # Shortcut preprocessing

            for data_directory in self.interim_directories:
                output_directory = determine_output_directory(
                    data_directory, self.setting.data.preprocessed_root,
                    self.str_replace)
                if not self.force_renew \
                        and self._file_exists(output_directory, variable_name):
                    print(
                        f"{output_directory} / {variable_name} "
                        'already exists. Skipped.')
                    continue

                util.copy_variable_file(
                    data_directory, variable_name, output_directory,
                    allow_missing=self.allow_missing)
            return

        for data_directory in self.interim_directories:
            output_directory = determine_output_directory(
                data_directory, self.setting.data.preprocessed_root,
                self.str_replace)
            if not self.force_renew \
                    and self._file_exists(output_directory, variable_name):
                print(
                    f"{output_directory} / {variable_name} "
                    'already exists. Skipped.')
                continue

            loaded_data = util.load_variable(
                data_directory, variable_name,
                allow_missing=self.allow_missing,
                decrypt_key=self.setting.data.encrypt_key)
            if loaded_data is None:
                continue
            else:
                transformed_data = preprocess_converter.transform(loaded_data)

            if self.save_func is None:
                util.save_variable(
                    output_directory, variable_name, transformed_data,
                    encrypt_key=self.setting.data.encrypt_key)
            else:
                self.save_func(
                    output_directory, variable_name, transformed_data)

        return


class Converter:

    def __init__(self, converter_parameters_pkl, key=None):
        self.converters = self._generate_converters(
            converter_parameters_pkl, key=key)
        return

    def _generate_converters(self, converter_parameters_pkl, key=None):
        if key is not None and converter_parameters_pkl.suffix == '.enc':
            return self._generate_converters(
                util.decrypt_file(key, converter_parameters_pkl))

        if isinstance(converter_parameters_pkl, io.BufferedIOBase):
            converter_parameters = pickle.load(converter_parameters_pkl)
        elif isinstance(converter_parameters_pkl, Path):
            with open(converter_parameters_pkl, 'rb') as f:
                converter_parameters = pickle.load(f)
        else:
            raise ValueError(
                f"Input type {converter_parameters_pkl.__class__} not "
                'understood')
        preprocess_setting = setting.PreprocessSetting(
            preprocess=converter_parameters)

        converters = {
            variable_name:
            util.PreprocessConverter(
                value['preprocess_converter'],
                method=value['method'],
                componentwise=value['componentwise'],
                other_components=value['other_components'])
            for variable_name, value in preprocess_setting.preprocess.items()}
        return converters

    def preprocess(self, dict_data_x):
        converted_dict_data_x = {
            variable_name:
            self.converters[variable_name].transform(data)
            for variable_name, data in dict_data_x.items()
            if variable_name in self.converters.keys()}
        return converted_dict_data_x

    def postprocess(
            self, dict_data_x, dict_data_y, output_directory=None, *,
            dict_data_y_answer=None,
            overwrite=False, save_x=False, write_simulation=False,
            write_npy=True, write_simulation_stem=None,
            write_simulation_base=None, read_simulation_type='fistr',
            save_function=None,
            write_simulation_type='fistr', skip_femio=False,
            load_function=None, convert_to_order1=False,
            data_addition_function=None, required_file_names=[],
            perform_inverse=True, **kwargs):
        """Postprocess data with inversely converting them.

        Parameters
        ----------
        dict_data_x: dict
            Dict of input data.
        dict_data_y: dict
            Dict of output data.
        output_directory: pathlib.Path, optional
            Output directory path.
        dict_data_y_answer: dict
            Dict of expected output data.
        overwrite: bool, optional
            If True, overwrite data.
        save_x: bool, optional
            If True, save input values in addition to output values.
        write_simulation: bool, optional
            If True, write simulation data file(s) based on the inference.
        write_npy: bool, optional
            If True, write npy files of inferences.
        write_simulation_base: pathlib.Path, optional
            Base of simulation data to be used for write_simulation option.
            If not fed, try to find from the input directories.
        read_simulation_type: str, optional
            Simulation file type to read simulation base.
        write_simulation_type: str, optional
            Simulation file type to write.
        skip_femio: bool, optional
            If True, skip femio to read simulation base.
        load_function: callable, optional
            Load function taking data_files and data_directory as inputs,
            and returns data_dict and fem_data.
        required_file_names: list[str], optional
            Required file names for load function.
        data_addition_function=callable, optional
            Function to add some data to existing fem_data.

        Returns
        --------
            inversed_dict_data_x: dict
                Inversed input data.
            inversed_dict_data_y: dict
                Inversed output data.
            fem_data: femio.FEMData
                FEMData object with input and output data.
        """
        if perform_inverse:
            dict_post_function = {
                k: v.inverse for k, v in self.converters.items()}
        else:
            dict_post_function = {
                k: lambda x: x for k, v in self.converters.items()}

        if isinstance(list(dict_data_x.values())[0], dict):
            return_dict_data_x = {
                variable_name:
                dict_post_function[variable_name](data)
                for value in dict_data_x.values()
                for variable_name, data in value.items()}
        else:
            return_dict_data_x = {
                variable_name:
                dict_post_function[variable_name](data)
                for variable_name, data in dict_data_x.items()}

        if dict_data_y_answer is not None and len(dict_data_y_answer) > 0:
            if isinstance(list(dict_data_y_answer.values())[0], dict):
                return_dict_data_x.update({
                    variable_name:
                    dict_post_function[variable_name](data)
                    for value in dict_data_y_answer.values()
                    for variable_name, data in value.items()})
            else:
                return_dict_data_x.update({
                    variable_name:
                    dict_post_function[variable_name](data)
                    for variable_name, data in dict_data_y_answer.items()})

        if len(dict_data_y) > 0:
            if isinstance(list(dict_data_y.values())[0], dict):
                return_dict_data_y = {
                    variable_name:
                    dict_post_function[variable_name](data)
                    for value in dict_data_y.values()
                    for variable_name, data in value.items()}
            else:
                return_dict_data_y = {
                    variable_name:
                    dict_post_function[variable_name](data)
                    for variable_name, data in dict_data_y.items()}
        else:
            return_dict_data_y = {}

        # Save data
        if write_simulation_base is None or not write_simulation_base.exists():
            fem_data = None
        else:
            try:
                fem_data = self._create_fem_data(
                    return_dict_data_x, return_dict_data_y,
                    write_simulation_base=write_simulation_base,
                    write_simulation_stem=write_simulation_stem,
                    read_simulation_type=read_simulation_type,
                    data_addition_function=data_addition_function,
                    skip_femio=skip_femio, load_function=load_function,
                    convert_to_order1=convert_to_order1,
                    required_file_names=required_file_names)
            except ValueError:
                fem_data = None
                write_simulation_base = None
                write_simulation = False
        if output_directory is not None:
            if write_npy:
                if save_x:
                    self.save(return_dict_data_x, output_directory)
                self.save(return_dict_data_y, output_directory)
            if write_simulation:
                if write_simulation_base is None:
                    raise ValueError('No write_simulation_base fed.')
                self._write_simulation(
                    output_directory, fem_data, overwrite=overwrite,
                    write_simulation_type=write_simulation_type)
            if save_function is not None:
                save_function(
                    output_directory, fem_data, overwrite=overwrite,
                    write_simulation_type=write_simulation_type)

        return return_dict_data_x, return_dict_data_y, fem_data

    def _create_fem_data(
            self, dict_data_x, dict_data_y, write_simulation_base, *,
            write_simulation_stem=None,
            read_simulation_type='fistr', data_addition_function=None,
            skip_femio=False, load_function=None,
            required_file_names=[], convert_to_order1=False):
        if not skip_femio:
            fem_data = femio.FEMData.read_directory(
                read_simulation_type, write_simulation_base,
                stem=write_simulation_stem, save=False)
        elif load_function:
            if len(required_file_names) == 0:
                raise ValueError(
                    'Please specify required_file_names when skip_femio '
                    'is True.')
            data_files = util.collect_files(
                write_simulation_base, required_file_names)
            data_dict, fem_data = load_function(
                data_files, write_simulation_base)
            fem_data = update_fem_data(
                fem_data, data_dict, allow_overwrite=True)
        else:
            raise ValueError(
                'When skip_femio is True, please feed load_function.')

        if convert_to_order1:
            fem_data = fem_data.to_first_order()

        fem_data = update_fem_data(fem_data, dict_data_x, prefix='answer_')
        fem_data = update_fem_data(fem_data, dict_data_y, prefix='inferred_')
        fem_data = add_difference(
            fem_data, dict_data_y, dict_data_x, prefix='difference_')
        if data_addition_function is not None:
            fem_data = data_addition_function(fem_data, write_simulation_base)

        return fem_data

    def _write_simulation(
            self, output_directory, fem_data, *,
            write_simulation_type='fistr', overwrite=False):
        if write_simulation_type == 'fistr':
            ext = ''
        elif write_simulation_type == 'ucd':
            ext = '.inp'
        elif write_simulation_type == 'vtk':
            ext = '.vtk'
        else:
            raise ValueError(
                f"Unexpected write_simulation_type: {write_simulation_type}")
        fem_data.write(
            write_simulation_type, output_directory / ('mesh' + ext),
            overwrite=overwrite)
        return

    def save(self, data_dict, output_directory):
        if not output_directory.exists():
            output_directory.mkdir(parents=True, exist_ok=True)
        for variable_name, data in data_dict.items():
            np.save(output_directory / f"{variable_name}.npy", data)
        return


def extract_variables(
        fem_data, mandatory_variables, *, optional_variables=None):
    """Extract variables from FEMData object to convert to data dictionary.

    Parameters
    ----------
    fem_data: femio.FEMData
        FEMData object to be extracted variables from.
    mandatory_variables: list[str]
        Mandatory variable names.
    optional_variables: list[str], optional
        Optional variable names.

    Returns
    -------
        dict_data: dict
            Data dictionary.
    """
    dict_data = {
        mandatory_variable: _extract_single_variable(
            fem_data, mandatory_variable, mandatory=True, ravel=True)
        for mandatory_variable in mandatory_variables}

    if optional_variables is not None and len(optional_variables) > 0:
        for optional_variable in optional_variables:
            optional_variable_data = _extract_single_variable(
                fem_data, optional_variable, mandatory=False, ravel=True)
            if optional_variable_data is not None:
                dict_data.update({optional_variable: optional_variable_data})
    return dict_data


def _extract_single_variable(
        fem_data, variable_name, *, mandatory=True, ravel=True):
    if variable_name in fem_data.nodal_data:
        return fem_data.convert_nodal2elemental(
            variable_name, ravel=ravel)
    elif variable_name in fem_data.elemental_data:
        return fem_data.elemental_data.get_attribute_data(variable_name)
    else:
        if mandatory:
            raise ValueError(
                f"{variable_name} not found in {fem_data.nodal_data.keys()}, "
                f"{fem_data.elemental_data.keys()}")
        else:
            return None


def save_dict_data(
        output_directory, dict_data, *, dtype=np.float32, encrypt_key=None,
        finished_file='converted'):
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
        util.save_variable(
            output_directory, key, value, dtype=dtype, encrypt_key=encrypt_key)
    (output_directory / finished_file).touch()
    return


def determine_output_directory(
        input_directory, output_base_directory, str_replace):
    """Determine output directory by replacing a string (str_replace) in the
    input_directory.

    Parameters
    ----------
    input_directory: pathlib.Path
        Input directory path.
    output_base_directory: pathlib.Path
        Output base directory path. The output directry name is under that
        directory.
    str_replace: str
        The string to be replaced.

    Returns
    -------
    output_directory: pathlib.Path
        Detemined output directory path.
    """
    common_prefix = Path(os.path.commonprefix(
        [input_directory, output_base_directory]))
    relative_input_path = Path(os.path.relpath(input_directory, common_prefix))
    parts = list(relative_input_path.parts)

    replace_indices = np.where(
        np.array(relative_input_path.parts) == str_replace)[0]
    if len(replace_indices) == 0:
        pass
    elif len(replace_indices) == 1:
        replace_index = replace_indices[0]
        parts[replace_index] = ''
    else:
        raise ValueError(
            f"Input directory {input_directory} contains several "
            f"{str_replace} parts thus ambiguous.")
    output_directory = output_base_directory / '/'.join(parts).lstrip('/')

    return output_directory


def normalize_adjacency_matrix(adj):
    """Symmetrically normalize adjacency matrix.

    Parameters
    ----------
    adj: scipy.sparse.coo_matrix
        Adjacency matrix in COO expression.

    Returns
    -------
    normalized_adj: scipy.sparse.coo_matrix
        Normalized adjacency matrix in COO expression.
    """
    print(f"to_coo adj: {dt.datetime.now()}")
    adj = sp.coo_matrix(adj)
    diag = adj.diagonal()
    additional_diag = np.zeros(len(diag))
    additional_diag[np.abs(diag) < 1.e-5] = 1.
    adj = adj + sp.diags(additional_diag)
    print(f"sum raw: {dt.datetime.now()}")
    rowsum = np.array(adj.sum(1))
    print(f"invert d: {dt.datetime.now()}")
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    print(f"making diag: {dt.datetime.now()}")
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    print(f"calculating norm: {dt.datetime.now()}")
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def analyze_data_directories(
        data_directories, x_names, f_names, *, n_split=10, n_bin=20,
        out_directory=None, ref_index=0, plot=True, symmetric=False,
        magnitude_range=1.):
    """Analyze data f_name with grid over x_name.

    Parameters
    ----------
    data_directories: list[pathlib.Path]
        List of data directories.
    x_names: list[str]
        Names of x variables.
    f_names: list[str]
        Name of f variable.
    n_split: int, optional
        The number to split x space.
    n_bin: int, optional
        The number of bins to draw histogram
    out_directory: pathlib.Path, optional
        Output directory path. By default no output is written.
    ref_index: int, optional
        Reference data directory index to analyze data.
    plot: bool, optional
        If True, plot data by grid.
    symmetric: bool, optional
        If True, take plot range symmetric.
    magnitude_range: float, optional
        Magnitude to be multiplied to the range of plot.
    """

    # Initialization
    if out_directory is not None:
        out_directory.mkdir(parents=True, exist_ok=True)

    data = [
        _read_analyzing_data(data_directory, x_names, f_names)
        for data_directory in data_directories]
    xs = [d[0] for d in data]
    fs = [d[1] for d in data]
    f_grids = _generate_grids(
        fs, n_bin, symmetric=symmetric, magnitude_range=magnitude_range)
    # f_grids = _generate_grids(fs, n_bin)

    ranges, list_split_data, centers, means, stds, coverage \
        = split_data_arrays(xs, fs, n_split=n_split)
    str_x_names = '-'.join(x_name for x_name in x_names)
    str_f_names = '-'.join(f_name for f_name in f_names)

    # Plot data
    if plot:
        for range_, split_data in zip(ranges, list_split_data):
            range_string = '__'.join(f"{r[0]:.3e}_{r[1]:.3e}" for r in range_)
            if out_directory is None:
                out_file_base = None
            else:
                out_file_base = out_directory / f"{str_f_names}_{range_string}"
            _plot_histogram(
                split_data, f_grids, data_directories,
                out_file_base=out_file_base)

    # Write output file
    array_means = np.transpose(np.stack(means), (1, 0, 2))
    mean_diffs = array_means - array_means[ref_index]
    array_stds = np.transpose(np.stack(stds), (1, 0, 2))
    std_diffs = array_stds - array_stds[ref_index]

    nonref_indices = list(range(ref_index)) + list(
        range(ref_index + 1, len(data)))
    nonref_mean_diffs = mean_diffs[nonref_indices]
    nonref_std_diffs = std_diffs[nonref_indices]

    mean_difference = np.mean(
        nonref_mean_diffs[~np.isnan(nonref_mean_diffs)]**2)**.5
    std_difference = np.mean(
        nonref_std_diffs[~np.isnan(nonref_std_diffs)]**2)**.5
    print(
        f"Mean difference: {mean_difference:.3e}\n"
        f" STD difference: {std_difference:.3e}\n"
        f"       Coverage: {coverage:.3f}")

    header = ','.join(
        f"{str_x_names}_{i}" for i in range(list(centers.shape)[-1])) + ',' \
        + ','.join(
            f"mean_diff_{str_f_names}_{i}"
            for i in range(mean_diffs.shape[-1])) \
        + ',mean_diff_norm,' \
        + ','.join(
            f"std_diff_{str_f_names}_{i}"
            for i in range(mean_diffs.shape[-1])) \
        + ',std_diff_norm'
    for i_dir, data_directory in enumerate(data_directories):
        mean_diff_norms = np.linalg.norm(mean_diffs[i_dir], axis=1)[:, None]
        std_diff_norms = np.linalg.norm(std_diffs[i_dir], axis=1)[:, None]
        if out_directory is not None:
            np.savetxt(
                out_directory / (data_directory.stem + '.csv'),
                np.concatenate(
                    [centers, mean_diffs[i_dir], mean_diff_norms,
                     std_diffs[i_dir], std_diff_norms], axis=1),
                delimiter=',', header=header)


def split_data_arrays(xs, fs, *, n_split=10, ref_index=0):
    """Split data fs with regards to grids of xs.

    Parameters
    ----------
    xs: list[numpy.ndarray]
        n_sample-length list contains (n_element, dim_x) shaped ndarray.
    fs: list[numpy.ndarray]
        n_sample-length list contains (n_element, dim_f) shaped ndarray.
    n_split: int, optional
        The number to split x space.
    """

    x_grids = _generate_grids(xs, n_split)

    # Analyze data by grid
    ranges = np.transpose(
        np.stack([x_grids[:-1, :], x_grids[1:, :]]), (2, 1, 0))
    useful_ranges = []
    list_split_data = []
    centers = []
    means = []
    stds = []
    n_cell_with_ref = 0
    for rs in it.product(*ranges):
        filters = [_calculate_filter(x, rs) for x in xs]
        if np.any(filters[ref_index]):
            n_cell_with_ref = n_cell_with_ref + 1
            if not np.any([
                    np.any(filter_) for filter_
                    in filters[:ref_index] + filters[ref_index+1:]]):
                continue
        else:
            continue

        filtered_fs = [f_[filter_] for f_, filter_ in zip(fs, filters)]
        list_split_data.append(filtered_fs)
        useful_ranges.append(rs)

        filtered_means = np.stack([
            np.mean(ff, axis=0) for ff in filtered_fs])
        filtered_stds = np.stack([
            np.std(ff, axis=0) for ff in filtered_fs])
        center = [np.mean(r) for r in rs]
        centers.append(center)
        means.append(filtered_means)
        stds.append(filtered_stds)
    coverage = len(useful_ranges) / n_cell_with_ref

    # Write output file
    centers = np.array(centers)

    return useful_ranges, list_split_data, centers, means, stds, coverage


def _plot_histogram(
        list_data, list_bins, data_directories,
        out_file_base=None):
    f_dim = list_data[0].shape[-1]
    plt.close('all')

    for i_dim in range(f_dim):
        plt.figure(i_dim)
        for data, data_directory in zip(list_data, data_directories):
            plt.hist(
                data[:, i_dim], bins=list_bins[:, i_dim],
                histtype='step', label=str(data_directory))

    if out_file_base is not None:
        for i_dim in range(f_dim):
            plt.figure(i_dim)
            plt.legend()
            plt.savefig(str(out_file_base) + f"_{i_dim}.pdf")
    else:
        for i_dim in range(f_dim):
            plt.figure(i_dim)
            plt.legend()
        plt.show()
    return


def _generate_grids(list_data, n_split, symmetric=False, magnitude_range=1.):
    bounding_box = _obtain_bounding_box(list_data) * magnitude_range
    if symmetric:
        bounding_box = np.stack([
            -np.mean(np.abs(bounding_box), axis=0),
            np.mean(np.abs(bounding_box), axis=0)])
    grids = np.linspace(bounding_box[0, :], bounding_box[1, :], n_split)
    return grids


def _calculate_filter(x, ranges):
    filter_ = np.ones(len(x))
    for _x, _r in zip(x.T, ranges):
        filter_ = np.logical_and(
            filter_, np.logical_and(_r[0] <= _x, _x < _r[1]))
    return filter_


def _obtain_bounding_box(data):
    concat_data = np.concatenate(data)

    # TODO: remove list() below after astroid is updated > 2.3.3
    return np.stack([
        [np.min(concat_data[:, i]), np.max(concat_data[:, i])]
        for i in range(list(concat_data.shape)[-1])], axis=1)


def _read_analyzing_data(data_directory, x_names, f_names):
    fem_data = femio.FEMData.read_directory('fistr', data_directory)
    for x_name in x_names:
        if x_name not in fem_data.elemental_data \
                and x_name not in fem_data.nodal_data:
            if x_name == 'node':
                fem_data.overwrite_attribute('NODE', fem_data.nodes.data)
                continue
            fem_data.elemental_data.update({
                x_name: femio.FEMAttribute(
                    x_name, fem_data.elements.ids,
                    np.load(data_directory / (x_name + '.npy')))})
    for f_name in f_names:
        if f_name not in fem_data.elemental_data \
                and f_name not in fem_data.nodal_data:
            if f_name == 'node':
                continue
            # fem_data.overwrite_attribute(
            #     f_name, np.load(data_directory / (f_name + '.npy')))
            fem_data.elemental_data.update({
                f_name: femio.FEMAttribute(
                    f_name, fem_data.elements.ids,
                    np.load(data_directory / (f_name + '.npy')))})

    x_val = np.concatenate([
        fem_data.convert_nodal2elemental(x_name, calc_average=True)
        for x_name in x_names], axis=-1)
    f_val = np.concatenate([
        fem_data.convert_nodal2elemental(f_name, calc_average=True)
        for f_name in f_names], axis=-1)
    return x_val, f_val

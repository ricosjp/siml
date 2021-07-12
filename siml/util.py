import datetime as dt
import gc
from glob import glob, iglob
import io
import os
from pathlib import Path
import re
import shutil
import subprocess

from Cryptodome.Cipher import AES

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
import yaml


INFERENCE_FLAG_FILE = 'inference'


def date_string():
    return dt.datetime.now().isoformat().replace('T', '_').replace(':', '-')


def load_yaml_file(file_name):
    """Load YAML file.

    Parameters
    ----------
    file_name: str or pathlib.Path
        YAML file name.

    Returns
    --------
    dict_data: dict
        YAML contents.
    """
    with open(file_name, 'r') as f:
        dict_data = yaml.load(f, Loader=yaml.SafeLoader)
    return dict_data


def load_yaml(source):
    """Load YAML source.

    Parameters
    ----------
    source: File-like object or str or pathlib.Path

    Returns
    --------
    dict_data: dict
        YAML contents.
    """
    if isinstance(source, io.TextIOBase):
        return yaml.load(source, Loader=yaml.SafeLoader)
    elif isinstance(source, str):
        return yaml.load(source, Loader=yaml.SafeLoader)
    elif isinstance(source, Path):
        return load_yaml_file(source)
    else:
        raise ValueError(f"Input type {source.__class__} not understood")


def save_variable(
        output_directory, file_basename, data,
        *, dtype=np.float32, encrypt_key=None):
    """Save variable data.

    Parameters
    ----------
    output_directory: pathlib.Path
        Save directory path.
    file_basename: str
        Save file base name without extenstion.
    data: np.ndarray or scipy.sparse.coo_matrix
        Data to be saved.
    dtype: type, optional
        Data type to be saved.
    encrypt_key: bytes, optional
        Data for encryption.

    Returns
    --------
        None
    """
    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
    if isinstance(data, np.ndarray):
        if encrypt_key is None:
            save_file_path = output_directory / (file_basename + '.npy')
            np.save(save_file_path, data.astype(dtype))
        else:
            save_file_path = output_directory / (file_basename + '.npy.enc')
            bytesio = io.BytesIO()
            np.save(bytesio, data.astype(dtype))
            encrypt_file(encrypt_key, save_file_path, bytesio)

    elif isinstance(data, (sp.coo_matrix, sp.csr_matrix, sp.csc_matrix)):
        if encrypt_key is None:
            save_file_path = output_directory / (file_basename + '.npz')
            sp.save_npz(save_file_path, data.tocoo().astype(dtype))
        else:
            save_file_path = output_directory / (file_basename + '.npz.enc')
            bytesio = io.BytesIO()
            sp.save_npz(bytesio, data.tocoo().astype(dtype))
            encrypt_file(encrypt_key, save_file_path, bytesio)
    else:
        raise ValueError(f"{file_basename} has unknown type: {data.__class__}")

    print(f"{file_basename} is saved in: {save_file_path}")
    return


def load_variable(
        data_directory, file_basename, *, allow_missing=False,
        check_nan=True, retry=True, decrypt_key=None):
    """Load variable data.

    Parameters
    ----------
    output_directory: pathlib.Path
        Directory path.
    file_basename: str
        File base name without extenstion.
    allow_missing: bool, optional
        If True, return None when the corresponding file is missing.
        Otherwise, raise ValueError.
    decrypt_key: bytes, optional
        If fed, it is used to decrypt the file.

    Returns
    --------
        data: numpy.ndarray or scipy.sparse.coo_matrix
    """
    if (data_directory / (file_basename + '.npy')).exists():
        loaded_data = np.load(data_directory / (file_basename + '.npy'))
        data_for_check_nan = loaded_data
        ext = '.npy'
    elif (data_directory / (file_basename + '.npy.enc')).exists():
        if decrypt_key is None:
            raise ValueError('Feed decrypt key')
        loaded_data = np.load(decrypt_file(
            decrypt_key, data_directory / (file_basename + '.npy.enc')))
        data_for_check_nan = loaded_data
        ext = '.npy.enc'
    elif (data_directory / (file_basename + '.npz')).exists():
        loaded_data = sp.load_npz(data_directory / (file_basename + '.npz'))
        data_for_check_nan = loaded_data.data
        ext = '.npz'
    elif (data_directory / (file_basename + '.npz.enc')).exists():
        if decrypt_key is None:
            raise ValueError('Feed decrypt key')
        loaded_data = sp.load_npz(decrypt_file(
            decrypt_key, data_directory / (file_basename + '.npz.enc')))
        data_for_check_nan = loaded_data.data
        ext = '.npz.enc'
    else:
        if allow_missing:
            return None
        else:
            if retry:
                print(f"Retrying for: {data_directory}")
                subprocess.run(
                    f"find {data_directory}", shell=True, check=True)
                loaded_data = load_variable(
                    data_directory, file_basename, allow_missing=allow_missing,
                    check_nan=check_nan, retry=False)
                return loaded_data
            else:
                raise ValueError(
                    'File type not understood or file missing for: '
                    f"{file_basename} in {data_directory}")

    if check_nan and np.any(np.isnan(data_for_check_nan)):
        raise ValueError(
            f"NaN found in {data_directory / (file_basename + ext)}")

    return loaded_data


def copy_variable_file(
        input_directory, file_basename, output_directory,
        *, allow_missing=False, retry=True):
    """Copy variable file.

    Parameters
    ----------
    input_directory: pathlib.Path
        Input directory path.
    file_basename: str
        File base name without extenstion.
    output_directory: pathlib.Path
        Putput directory path.
    allow_missing: bool, optional
        If True, return None when the corresponding file is missing.
        Otherwise, raise ValueError.
    """
    if (input_directory / (file_basename + '.npy')).exists():
        ext = '.npy'
    elif (input_directory / (file_basename + '.npy.enc')).exists():
        ext = '.npy.enc'
    elif (input_directory / (file_basename + '.npz')).exists():
        ext = '.npz'
    elif (input_directory / (file_basename + '.npz.enc')).exists():
        ext = '.npz.enc'
    else:
        if allow_missing:
            return
        else:
            if retry:
                print(f"Retrying for: {input_directory}")
                subprocess.run(
                    f"find {input_directory}", shell=True, check=True)
                copy_variable_file(
                    input_directory, file_basename, output_directory,
                    allow_missing=allow_missing, retry=False)
                return
            else:
                raise ValueError(
                    'File type not understood or file missing for: '
                    f"{file_basename} in {input_directory}")
    basename = file_basename + ext
    output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        input_directory / basename, output_directory / basename)

    return


def collect_data_directories(
        base_directory, *, required_file_names=None, allow_no_data=False,
        pattern=None, inverse_pattern=None, toplevel=True):
    """Collect data directories recursively from the base directory.

    Parameters
    ----------
    base_directory: pathlib.Path
        Base directory to search directory from.
    required_file_names: list[str]
        If given, return only directories which have required files.
    pattern: str
        If given, return only directories which match the pattern.
    inverse_pattern: str, optional
        If given, return only files which DO NOT match the pattern.

    Returns
    --------
    found_directories: list[pathlib.Path]
        All found directories.
    """
    if isinstance(base_directory, (list, tuple, set)):
        found_directories = list(np.unique(np.concatenate([
            collect_data_directories(
                bd, required_file_names=required_file_names,
                allow_no_data=allow_no_data, pattern=pattern,
                inverse_pattern=inverse_pattern, toplevel=False)
            for bd in base_directory])))
        found_directories = _validate_found_directories(
            base_directory, found_directories, pattern, inverse_pattern,
            allow_no_data)
        return found_directories

    str_base_directory = str(base_directory).rstrip('/') + '/**'
    found_directories = iglob(str_base_directory, recursive=True)
    if required_file_names:
        found_directories = [
            Path(g) for g in found_directories
            if Path(g).is_dir()
            and directory_have_files(Path(g), required_file_names)]
    else:
        found_directories = [
            Path(g) for g in found_directories
            if Path(g).is_dir()]

    if toplevel:
        found_directories = _validate_found_directories(
            base_directory, found_directories, pattern, inverse_pattern,
            allow_no_data)

    return found_directories


def _validate_found_directories(
        base_directory, found_directories, pattern, inverse_pattern,
        allow_no_data):
    if pattern is not None:
        found_directories = [
            d for d in found_directories if re.search(pattern, str(d))]

    if inverse_pattern is not None:
        found_directories = [
            d for d in found_directories
            if not re.search(inverse_pattern, str(d))]

    if not allow_no_data and len(found_directories) == 0:
        raise ValueError(f"No data found in {base_directory}")

    return found_directories


def directory_have_files(directory, files):
    if isinstance(files, str):
        files = [files]
    return np.all([len(glob(str(directory / f) + '*')) > 0 for f in files])


def collect_files(
        directories, required_file_names, *, pattern=None,
        allow_no_data=False, inverse_pattern=None):
    """Collect data files recursively from the base directory.

    Parameters
    ----------
    base_directory: pathlib.Path or list[pathlib.Path]
        Base directory to search directory from.
    required_file_names: list[str]
        File names.
    pattern: str, optional
        If given, return only files which match the pattern.
    inverse_pattern: str, optional
        If given, return only files which DO NOT match the pattern.

    Returns
    -------
    collected_files: list[pathlib.Path]
    """
    if isinstance(required_file_names, list):
        found_files = []
        for required_file_name in required_file_names:
            found_files = found_files + collect_files(
                directories, required_file_name, pattern=pattern,
                inverse_pattern=inverse_pattern)
        return found_files

    if isinstance(directories, list):
        return list(np.unique(np.concatenate([
            collect_files(
                d, required_file_names, pattern=pattern,
                inverse_pattern=inverse_pattern)
            for d in directories])))

    required_file_name = required_file_names
    found_files = glob(
        str(directories / f"**/{required_file_name}"), recursive=True)

    if pattern is not None:
        found_files = [
            f for f in found_files if re.search(pattern, str(f))]

    if inverse_pattern is not None:
        found_files = [
            f for f in found_files
            if not re.search(inverse_pattern, str(f))]

    if not allow_no_data and len(found_files) == 0:
        message = f"No files found for {required_file_names} in {directories}"
        if pattern is not None:
            message = message + f"with pattern {pattern}"
        raise ValueError(message)

    return found_files


def files_match(file_names, required_file_names):
    """Check if file names match.

    Parameters
    ----------
    file_names: list[str]
    file_names: list[str]

    Returns
    --------
    files_match: bool
        True if all files match. Otherwise False.
    """
    replaced_required_file_names = [
        required_file_name.replace('.', r'\.').replace('*', '.*')
        for required_file_name in required_file_names]
    return np.all([
        np.any([
            re.search(replaced_required_file_name, file_name)
            for file_name in file_names])
        for replaced_required_file_name in replaced_required_file_names])


def files_exist(directory, file_names):
    """Check if files exist in the specified directory.

    Parameters
    ----------
    directory: pathlib.Path
    file_names: list[str]

    Returns
    --------
    files_exist: bool
        True if all files exist. Otherwise False.
    """
    if isinstance(file_names, str):
        file_names = [file_names]
    a = np.all([
        len(list(directory.glob(file_name))) > 0
        for file_name in file_names])
    return a


class PreprocessConverter():

    MAX_RETRY = 3

    def __init__(
            self, setting_data, *,
            data_files=None, componentwise=True, power=1., method=None,
            other_components=[], key=None):
        self.is_erroneous = None
        self.setting_data = setting_data
        self.power = power
        self.other_components = other_components
        self.key = key
        self.use_diagonal = False
        self.method = method

        self._init_converter()

        self.componentwise = componentwise
        self.retry_count = 0

        if not isinstance(
                self.converter, (SparseStandardScaler, MaxAbsScaler)):
            if abs(self.power - 1) > 1e-5:
                raise ValueError(
                    f"power option is not supported for {self.converter}")

        if data_files is not None:
            self.lazy_read_files(data_files)
        return

    def _init_converter(self):
        if isinstance(self.setting_data, dict):
            self._init_with_dict(self.setting_data)
        elif isinstance(self.setting_data, str):
            self._init_with_str(self.setting_data)
        elif isinstance(self.setting_data, BaseEstimator):
            self._init_with_converter(self.setting_data)
        elif isinstance(self.setting_data, PreprocessConverter):
            self._init_with_converter(self.setting_data.converter)
        else:
            raise ValueError(f"Unsupported setting_data: {self.setting_data}")

    def _init_with_dict(self, setting_dict):
        if 'method' in setting_dict:
            preprocess_method = setting_dict['method']
            self._init_with_str(preprocess_method)
        else:
            if self.method is None:
                raise ValueError('Feed ''method'' when initialize with pkl')
            self._init_with_str(self.method)
            for key, value in setting_dict.items():
                setattr(self.converter, key, value)
        return

    def _init_with_str(self, preprocess_method):
        if preprocess_method == 'identity':
            self.converter = Identity()
        elif preprocess_method == 'standardize':
            self.converter = preprocessing.StandardScaler()
            self.is_erroneous = self.is_standard_scaler_var_nan
        elif preprocess_method == 'std_scale':
            self.converter = preprocessing.StandardScaler(with_mean=False)
            self.is_erroneous = self.is_standard_scaler_var_nan
        elif preprocess_method == 'sparse_std':
            self.converter = SparseStandardScaler(
                power=self.power, other_components=self.other_components)
            self.is_erroneous = self.is_standard_scaler_var_nan
        elif preprocess_method == 'isoam_scale':
            self.converter = IsoAMScaler(
                other_components=self.other_components)
            self.is_erroneous = self.is_standard_scaler_var_nan
            self.use_diagonal = True
        elif preprocess_method == 'min_max':
            self.converter = preprocessing.MinMaxScaler()
        elif preprocess_method == 'max_abs':
            self.converter = MaxAbsScaler(power=self.power)
        else:
            raise ValueError(
                f"Unknown preprocessing method: {preprocess_method}")
        return

    def _init_with_converter(self, converter):
        self.converter = converter
        return

    def apply_data_with_rehspe_if_needed(
            self, data, function, return_applied=True, use_diagonal=False):
        if isinstance(data, np.ndarray):
            if use_diagonal:
                raise ValueError('Cannot set use_diagonal=True for dense data')
            result = self.apply_numpy_data_with_reshape_if_needed(
                data, function, return_applied=return_applied)
        elif isinstance(data, (sp.coo_matrix, sp.csr_matrix, sp.csc_matrix)):
            result = self.apply_sparse_data_with_reshape_if_needed(
                data, function, return_applied=return_applied,
                use_diagonal=use_diagonal)
        else:
            raise ValueError(f"Unsupported data type: {data.__class__}")

        return result

    def is_standard_scaler_var_nan(self):
        return np.any(np.isnan(self.converter.var_))

    def apply_sparse_data_with_reshape_if_needed(
            self, data, function, return_applied=True, use_diagonal=False):
        if self.componentwise:
            applied = function(data)
            if return_applied:
                return applied.tocoo()
            else:
                return

        elif use_diagonal:
            print('Start diagonal')
            print(dt.datetime.now())
            reshaped = data.diagonal()
            print('Start apply')
            print(dt.datetime.now())
            applied_reshaped = function(reshaped)
            if return_applied:
                raise ValueError(
                    'Cannot set return_applied=True when use_diagonal=True')
            else:
                return

        else:
            shape = data.shape
            print('Start reshape')
            print(dt.datetime.now())
            reshaped = data.reshape((shape[0] * shape[1], 1))
            print('Start apply')
            print(dt.datetime.now())
            applied_reshaped = function(reshaped)
            if return_applied:
                return applied_reshaped.reshape(shape).tocoo()
            else:
                return

    def apply_numpy_data_with_reshape_if_needed(
            self, data, function, return_applied=True):
        shape = data.shape

        if self.componentwise:
            if len(shape) == 2:
                applied = function(data)
                if return_applied:
                    return applied
                else:
                    return
            elif len(shape) == 3:
                # Time series
                reshaped = np.reshape(data, (shape[0] * shape[1], shape[2]))
                applied_reshaped = function(reshaped)
                if return_applied:
                    applied = np.reshape(applied_reshaped, shape)
                    return applied
                else:
                    return
            elif len(shape) == 4:
                # Batched time series
                reshaped = np.reshape(
                    data, (shape[0] * shape[1] * shape[2], shape[3]))
                applied_reshaped = function(reshaped)
                if return_applied:
                    applied = np.reshape(applied_reshaped, shape)
                    return applied
                else:
                    return
            else:
                raise ValueError(f"Data shape {data.shape} not understood")

        else:
            reshaped = np.reshape(data, (-1, 1))
            applied_reshaped = function(reshaped)
            if return_applied:
                applied = np.reshape(applied_reshaped, shape)
                return applied
            else:
                return

    def lazy_read_files(self, data_files):
        for data_file in data_files:
            print(f"Start load data: {data_file}")
            print(dt.datetime.now())
            data = self.load_file(data_file)
            print(f"Start partial_fit: {data_file}")
            print(dt.datetime.now())
            self.apply_data_with_rehspe_if_needed(
                data, self.converter.partial_fit, return_applied=False,
                use_diagonal=self.use_diagonal)
            print(f"Start del: {data_file}")
            print(dt.datetime.now())
            del data
            print(f"Start GC: {data_file}")
            print(dt.datetime.now())
            gc.collect()
            print(f"Finish one iter: {data_file}")
            print(dt.datetime.now())

        if self.is_erroneous is not None:
            # NOTE: Check varianve is not none for StandardScaler with sparse
            # data. Related to
            # https://github.com/scikit-learn/scikit-learn/issues/16448
            if self.is_erroneous():
                if self.retry_count < self.MAX_RETRY:
                    print(
                        f"Retry for {data_file.stem}: {self.retry_count + 1}")
                    self.retry_count = self.retry_count + 1
                    np.random.shuffle(data_files)
                    self._init_converter()
                    self.lazy_read_files(data_files)
                else:
                    raise ValueError('Retry exhausted. Check the data.')

        return

    def load_file(self, data_file):
        str_data_file = str(data_file)
        if str_data_file.endswith('.npy'):
            data = np.load(data_file)
        elif str_data_file.endswith('.npy.enc'):
            data = np.load(decrypt_file(self.key, data_file))
        elif str_data_file.endswith('.npz'):
            data = sp.load_npz(data_file)
            if not sp.issparse(data):
                raise ValueError(f"Data type not understood for: {data_file}")
        elif str_data_file.endswith('.npz.enc'):
            data = sp.load_npz(decrypt_file(self.key, data_file))
            if not sp.issparse(data):
                raise ValueError(f"Data type not understood for: {data_file}")
        else:
            raise ValueError(f"Data type not understood for: {data_file}")
        return data

    def transform(self, data):
        return self.apply_data_with_rehspe_if_needed(
            data, self.converter.transform)

    def inverse(self, data):
        return self.apply_data_with_rehspe_if_needed(
            data, self.converter.inverse_transform)


class MaxAbsScaler(TransformerMixin, BaseEstimator):

    def __init__(self, power=1.):
        self.max_ = 0.
        self.power = power
        return

    def partial_fit(self, data):
        if sp.issparse(data):
            self.max_ = np.maximum(
                np.ravel(np.max(np.abs(data), axis=0).toarray()), self.max_)
        else:
            self.max_ = np.maximum(
                np.max(np.abs(data), axis=0), self.max_)
        return self

    def transform(self, data):
        if np.max(self.max_) == 0.:
            scale = 0.
        else:
            scale = (1 / self.max_)**self.power

        if sp.issparse(data):
            if len(scale) != 1:
                raise ValueError('Should be componentwise: false')
            scale = scale[0]
        return data * scale

    def inverse_transform(self, data):
        inverse_scale = self.max_
        if sp.issparse(data):
            if len(inverse_scale) != 1:
                raise ValueError('Should be componentwise: false')
            inverse_scale = inverse_scale[0]**(self.power)
        return data * inverse_scale


class SparseStandardScaler(TransformerMixin, BaseEstimator):
    """Class to perform standardization for sparse data."""

    def __init__(self, power=1., other_components=[]):
        self.var_ = 0.
        self.std_ = 0.
        self.mean_square_ = 0.
        self.n_ = 0
        self.power = power
        self.component_dim = len(other_components) + 1
        return

    def partial_fit(self, data):
        self._raise_if_not_sparse(data)
        self._update(data)
        return self

    def _update(self, sparse_dats):
        m = np.prod(sparse_dats.shape)
        mean_square = (
            self.mean_square_ * self.n_ + np.sum(sparse_dats.data**2)) / (
                self.n_ + m)

        self.mean_square_ = mean_square
        self.n_ += m

        # To use mean_i [x_i^2 + y_i^2 + z_i^2], multiply by the dim
        self.var_ = self.mean_square_ * self.component_dim
        self.std_ = np.sqrt(self.var_)
        return

    def _raise_if_not_sparse(self, data):
        if not sp.issparse(data):
            raise ValueError('Data is not sparse')
        return

    def transform(self, data):
        self._raise_if_not_sparse(data)
        if self.std_ == 0.:
            scale = 0.
        else:
            scale = (1 / self.std_)**self.power
        return data * scale

    def inverse_transform(self, data):
        self._raise_if_not_sparse(data)
        inverse_scale = self.std_**(self.power)
        return data * inverse_scale


class IsoAMScaler(TransformerMixin, BaseEstimator):
    """Class to perform scaling for IsoAM based on
    https://arxiv.org/abs/2005.06316.
    """

    def __init__(self, other_components=[]):
        self.var_ = 0.
        self.std_ = 0.
        self.mean_square_ = 0.
        self.n_ = 0
        self.component_dim = len(other_components) + 1
        if self.component_dim == 1:
            raise ValueError(
                'To use IsoAMScaler, feed other_components: '
                f"{other_components}")
        return

    def partial_fit(self, data):
        self._update(data)
        return self

    def _update(self, diagonal_data):
        if len(diagonal_data.shape) != 1:
            raise ValueError(f"Input data should be 1D: {diagonal_data}")
        m = len(diagonal_data)
        mean_square = (
            self.mean_square_ * self.n_ + np.sum(diagonal_data**2)) / (
                self.n_ + m)

        self.mean_square_ = mean_square
        self.n_ += m

        # To use mean_i [x_i^2 + y_i^2 + z_i^2], multiply by the dim
        self.var_ = self.mean_square_ * self.component_dim
        self.std_ = np.sqrt(self.var_)
        return

    def _raise_if_not_sparse(self, data):
        if not sp.issparse(data):
            raise ValueError('Data is not sparse')
        return

    def transform(self, data):
        if self.std_ == 0.:
            scale = 0.
        else:
            scale = (1 / self.std_)
        return data * scale

    def inverse_transform(self, data):
        self._raise_if_not_sparse(data)
        inverse_scale = self.std_
        return data * inverse_scale


class Identity(TransformerMixin, BaseEstimator):
    """Class to perform identity conversion (do nothing)."""

    def partial_fit(self, data):
        return

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


def get_top_directory():
    completed_process = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        capture_output=True, text=True)
    path = Path(completed_process.stdout.rstrip('\n'))
    return path


def pad_array(array, n):
    """Pad array to the size n.

    Parameters
    ----------
    array: numpy.ndarray or scipy.sparse.coo_matrix
        Input array of size (m, f1, f2, ...) for numpy.ndarray or (m. m)
        for scipy.sparse.coomatrix
    n: int
        Size after padding. n should be equal to or larger than m.

    Returns
    --------
    padded_array: numpy.ndarray or scipy.sparse.coo_matrix
        Padded array of size (n, f1, f2, ...) for numpy.ndarray or (n, n)
        for scipy.sparse.coomatrix.
    """
    shape = array.shape
    residual_length = n - shape[0]
    if residual_length < 0:
        raise ValueError('Max length of element is wrong.')
    if isinstance(array, np.ndarray):
        return np.concatenate(
            [array, np.zeros([residual_length] + list(shape[1:]))])
    elif sp.isspmatrix_coo(array):
        return sp.coo_matrix(
            (array.data, (array.row, array.col)), shape=(n, n))
    else:
        raise ValueError(f"Unsupported data type: {array.__class__}")


def concatenate_variable(variables):
    concatenatable_variables = np.concatenate(
        [
            _to_atleast_2d(variable) for variable in variables
            if isinstance(variable, np.ndarray)],
        axis=-1)
    unconcatenatable_variables = [
        variable for variable in variables
        if not isinstance(variable, np.ndarray)]
    if len(unconcatenatable_variables) == 0:
        return concatenatable_variables
    else:
        return concatenatable_variables, unconcatenatable_variables


def _to_atleast_2d(array):
    shape = array.shape
    if len(shape) == 1:
        return array[:, None]
    else:
        return array


def determine_max_process(max_process=None):
    """Determine maximum number of processes.

    Parameters
    ----------
    max_process: int, optional
        Input maximum process.

    Returns
    -------
    resultant_max_process: int
    """
    if hasattr(os, 'sched_getaffinity'):
        # This is more accurate in the cluster
        available_max_process = len(os.sched_getaffinity(0))
    else:
        available_max_process = os.cpu_count()
    if max_process is None:
        resultant_max_process = available_max_process
    else:
        resultant_max_process = min(available_max_process, max_process)
    return resultant_max_process


def split_data(list_directories, *, validation=.1, test=.1, shuffle=True):
    """Split list of data directories into train, validation, and test.

    Parameters
    ----------
    list_directories: list[pathlib.Path]
        List of data directories.
    validation: float, optional
        The ratio of the validation dataset size.
    test: float, optional
        The ratio of the test dataset size.
    shuffle: bool, optional
        If True, shuffle list_directories.

    Returns
    -------
    train_directories: list[pathlib.Path]
    validation_directories: list[pathlib.Path]
    test_directories: list[pathlib.Path]
    """
    if validation + test > 1.:
        raise ValueError(
            f"Sum of validation + test should be < 1. but {validation+test}")
    if shuffle:
        np.random.shuffle(list_directories)
    list_directories = np.asarray(list_directories)

    data_length = len(list_directories)
    if validation < 1e-5:
        validation_length = 0
    else:
        validation_length = int(np.floor(data_length * validation))

    if test < 1e-5:
        test_length = 0
    else:
        test_length = int(np.floor(data_length * test))

    validation_directories = list_directories[:validation_length]
    test_directories = list_directories[
        validation_length:validation_length+test_length]
    train_directories = list_directories[validation_length+test_length:]

    return train_directories, validation_directories, test_directories


def encrypt_file(key, file_path, binary):
    """Encrypt data and then save to a file.

    Parameters
    ----------
    key: bytes
        Key for encription.
    file_path: str or pathlib.Path
        File path to save.
    binary: io.BytesIO
        Data content.
    """
    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(binary.getvalue())
    with open(file_path, "wb") as f:
        [f.write(x) for x in (cipher.nonce, tag, ciphertext)]


def decrypt_file(key, file_name, return_stringio=False):
    """Decrypt data file.

    Parameters
    ----------
    key: bytes
        Key for decryption.
    file_path: str or pathlib.Path
        File path of the encrypted data.
    return_stringio: bool, optional
        If True, return io.StrintIO instead of io.BytesIO.

    Returns
    -------
    decrypted_data: io.BytesIO
    """
    with open(file_name, "rb") as f:
        nonce, tag, ciphertext = [f.read(x) for x in (16, 16, -1)]
    cipher = AES.new(key, AES.MODE_EAX, nonce)
    if return_stringio:
        return cipher.decrypt_and_verify(ciphertext, tag).decode('utf-8')
    else:
        return io.BytesIO(cipher.decrypt_and_verify(ciphertext, tag))

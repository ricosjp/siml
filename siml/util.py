import datetime as dt
from glob import glob, iglob
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import List

from Cryptodome.Cipher import AES

import numpy as np
import scipy.sparse as sp
import torch
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
        check_nan=False, retry=True, decrypt_key=None):
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
        pattern=None, inverse_pattern=None, toplevel=True, print_state=False):
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
    print_state: bool, optional
        If True, print state of the search

    Returns
    --------
    found_directories: list[pathlib.Path]
        All found directories.
    """
    if print_state:
        print(f"Searching: {base_directory}")

    if isinstance(base_directory, (list, tuple, set)):
        found_directories = list(np.unique(np.concatenate([
            collect_data_directories(
                bd, required_file_names=required_file_names,
                allow_no_data=allow_no_data, pattern=pattern,
                inverse_pattern=inverse_pattern, toplevel=False,
                print_state=print_state)
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
                inverse_pattern=inverse_pattern, allow_no_data=True)
        return found_files

    if isinstance(directories, list):
        return list(np.unique(np.concatenate([
            collect_files(
                d, required_file_names, pattern=pattern,
                inverse_pattern=inverse_pattern, allow_no_data=True)
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


class VariableMask:

    def __init__(self, skips, dims, is_dict=None, *, invert=False):
        if invert:
            skips = self._invert(skips)

        if is_dict is None:
            is_dict = isinstance(skips, dict)
        if isinstance(skips, list):
            if not np.any(skips):
                self.mask_function = self._identity_mask
                return
        elif isinstance(skips, dict):
            pass
            # NOTE: Not using _dict_identity_mask in case the output has
            #       assitional keys.
            # if np.all([not np.any(v) for v in skips.values()]):
            #     self.mask_function = self._dict_identity_mask
            #     self.mask = {
            #         key: self._generate_mask(skip_value, dims[key])
            #         for key, skip_value in skips.items()}
            #     return
        else:
            raise NotImplementedError

        print(f"skips: {skips}")
        if is_dict:
            self.mask = {
                key: self._generate_mask(skip_value, dims[key])
                for key, skip_value in skips.items()}
            self.mask_function = self._dict_mask
        else:
            self.mask = self._generate_mask(skips, dims)
            self.mask_function = self._array_mask

        return

    def __call__(self, *xs, **kwarg):
        return self.mask_function(*xs, **kwarg)

    def _invert(self, skips):
        if isinstance(skips, list):
            return [not s for s in skips]
        elif isinstance(skips, dict):
            return {k: [not s for s in v] for k, v in skips.items()}

    def _generate_mask(self, skips, dims):
        return ~np.array(np.concatenate([
            [skip] * dim for skip, dim in zip(skips, dims)]))

    def _identity_mask(self, *xs):
        return xs

    def _dict_identity_mask(self, *xs, keep_empty_data=None):
        return [
            [x[key] for key in xs[0].keys()] for x in xs]

    def _dict_mask(self,
                   *xs,
                   keep_empty_data=True,
                   with_key_names=False):
        masked_tensors = self._calc_masked_tensors(*xs)

        if keep_empty_data:
            tensors = self._replace_zero_element_tensor(masked_tensors)
        else:
            tensors = self._remove_zero_element_tensor(masked_tensors)

        if with_key_names:
            keys = self._get_masked_key_names(*xs)
            return *tensors, keys
        else:
            return tensors

    def _array_mask(self, *xs):
        return [x[..., self.mask] for x in xs]

    def _calc_masked_tensors(self, *xs):
        try:
            masked = [
                [
                    x[key][..., self.mask[key]]
                    for key in self.mask.keys() if key in x
                ]
                for x in xs
            ]
            return masked
        except IndexError as e:
            x = xs[0]
            raise ValueError(f"{e}\n", {
                key: (x[key].shape, self.mask[key].shape)
                for key in self.mask.keys() if key in x}
            )

    def _get_masked_key_names(self, *xs):
        masked_keys = [
            [key for key in self.mask.keys() if key in x]
            for x in xs
        ]

        # check whether key names are same
        for i in range(len(masked_keys) - 1):
            if masked_keys[i] != masked_keys[i + 1]:
                raise Exception(f"Key names are not matched."
                                f"{i}: {masked_keys[i]},"
                                f" {i + 1}: {masked_keys[i+1]}")

        return masked_keys[0]

    def _remove_zero_element_tensor(self,
                                    tensors: List[torch.Tensor]):
        return [
            [m_ for m_ in m if torch.numel(m_) > 0]
            for m in tensors
        ]

    def _replace_zero_element_tensor(self,
                                     tensors: List[torch.Tensor]):
        return [
            [
                torch.zeros(1).to(m_.device)
                if torch.numel(m_) == 0 else m_ for m_ in m
            ]
            for m in tensors
        ]


def cat_time_series(x, time_series_keys):

    if isinstance(x[0], dict):
        len_x = len(x)
        return {
            k: torch.cat([x[i][k] for i in range(len_x)], dim=1)
            if k in time_series_keys
            else torch.cat([x[i][k] for i in range(len_x)], dim=0)
            for k in x[0].keys()}
    else:
        return torch.cat(x, dim=1)  # Assume all are time series

"""Module for preprocessing."""

import datetime as dt
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import femio
from . import util


DTYPE = np.float32
FEMIO_FILE = 'femio_npy_saved.npy'


def convert_raw_data(
        raw_directory, mandatory_variables, *, optional_variables,
        output_base_directory='data/interim',
        recursive=False, conversion_function=None, force_renew=False,
        finished_file='converted', file_type='fistr',
        required_file_names=['*.msh', '*.cnt', '*.res.0.1'], read_npy=False):
    """Convert raw data and save them in interim directory.

    Args:
        raw_directory: str or pathlib.Path or list of them
            Raw data directory name.
        mandatory_variables: list of str
            Mandatory variable names. If any of them are not found,
            ValueError is raised.
        optional_variables: list of str
            Optional variable names. If any of them are not found,
            they are ignored.
        output_base_directory: str or pathlib.Path, optional ['data/interim']
            Output base directory for the converted raw data. By default,
            'data/interim' is the output base directory, so
            'data/interim/aaa/bbb' directory is the output directory for
            'data/raw/aaa/bbb' directory.
        recursive: bool, optional [False]
            If True, recursively convert data.
        conversion_function: function, optional [None]
            Conversion function which takes femio.FEMData object as an only
            argument and returns data dict to be saved.
        force_renew: bool, optional [False]
            If True, renew npy files even if they are alerady exist.
        finished_file: str, optional ['converted']
            File name to indicate that the conversion is finished.
        file_type: str, optional ['fistr']
            File type to be read.
        required_file_names: list of str,
                optional [['*.msh', '*.cnt', '*.res.0.1']]
            Required file names.
        read_npy: bool, optional [False]
            If True, read .npy files instead of original files if exists.
    Returns:
        None
    """

    # Process all directories when raw directory is a list
    if isinstance(raw_directory, list) or isinstance(raw_directory, set):
        for _raw_directory in raw_directory:
            convert_raw_data(
                _raw_directory, mandatory_variables,
                optional_variables=optional_variables,
                output_base_directory=output_base_directory,

                recursive=recursive,
                conversion_function=conversion_function,
                force_renew=force_renew, finished_file=finished_file,
                file_type=file_type,
                required_file_names=required_file_names, read_npy=read_npy)
        return

    # Process all subdirectories when recursice is True
    raw_directory = Path(raw_directory)
    output_base_directory = Path(output_base_directory)
    if recursive:
        raw_directories = util.collect_data_directories(
            raw_directory)
        convert_raw_data(
            raw_directories, mandatory_variables,
            optional_variables=optional_variables,
            output_base_directory=output_base_directory,
            recursive=recursive,
            conversion_function=conversion_function,
            force_renew=force_renew, finished_file=finished_file,
            file_type=file_type,
            required_file_names=required_file_names,
            read_npy=read_npy)

    # Determine output directory
    output_directory = determine_output_directory(
        raw_directory, output_base_directory, 'raw')

    # Guard
    if not util.files_exist(raw_directory, required_file_names):
        return
    if (output_directory / finished_file).exists() and not force_renew:
        print(f"Already converted. Skipped conversion: {raw_directory}")
        return

    # Main process
    if read_npy and (output_directory / FEMIO_FILE).exists():
        fem_data = femio.FEMData.read_npy_directory(output_directory)
    else:
        fem_data = femio.FEMData.read_directory(
            file_type, raw_directory, read_npy=read_npy, save=False)

    dict_data = extract_variables(
        fem_data, mandatory_variables, optional_variables=optional_variables)
    if conversion_function is not None:
        dict_data.update(conversion_function(fem_data))

    # Save data
    fem_data.save(output_directory)
    save_dict_data(output_directory, dict_data)
    (output_directory / finished_file).touch()

    return


def preprocess_interim_data(
        interim_directory, preprocess_methods, *,
        output_base_directory='data/preprocessed',
        force_renew=False, finished_file='preprocessed',
        required_file_names=['converted']):
    """Preprocess interim data with preprocessing e.g. standardization and then
    save them.

    Args:
        interim_directory: str or pathlib.Path
            Input base directory.
        preprocess_methods: dict
            Dict of variable name to preprocess method.
        output_base_directory: str or pathlib.Path,
                optional ['data/preprocessed']
            Output base directory.
        force_renew: bool, optional [False]
            If True, renew npy files even if they are alerady exist.
        finished_file: str, optional ['preprocessed']
            File name to indicate that the preprocessing is finished.
        required_file_names: list of str, optional [['converted']]
            Required file names.
    Returns:
        None
    """
    interim_directory = Path(interim_directory)
    output_base_directory = Path(output_base_directory)
    interim_directories = util.collect_data_directories(
        interim_directory, required_file_names=required_file_names)

    # Preprocess data variable by variable
    for variable_name, preprocess_method in preprocess_methods.items():
        preprocess_single_variable(
            interim_directory, output_base_directory, interim_directories,
            variable_name, preprocess_method, str_replace='interim',
            finished_file=finished_file, force_renew=force_renew)

    return


def preprocess_single_variable(
        data_base_directory, output_base_directory, data_directories,
        variable_name, preprocess_method, *, str_replace='interim',
        finished_file='preprocessed', force_renew=False):
    """Preprocess single variable.

    Args:
        data_base_directory: pathlib.Path
            Input base directory.
        output_base_directory: pathlib.Path
            Output base directory.
        data_directories: list of pathlib.Path
            Data directories.
        variable_name: str
            The name of the variable.
        preprocess_method: str
            Preprocess method name.
        str_replace: str, optional ['interim']
            Name to replace the input data base directory with.
        finished_file: str, optional ['preprocessed']
            File name to indicate that the preprocessing is finished.
        force_renew: bool, optional [False]
            If True, renew npy files even if they are alerady exist.
    Returns:
        None
    """

    # Check if data already exists
    if not force_renew and np.any([
            util.files_exist(
                determine_output_directory(
                    data_directory, output_base_directory, str_replace),
                [variable_name + '.*'])
            for data_directory in data_directories]):
        print(f"Data already exists in {output_base_directory}. Skipped.")
        exit()

    # Prepare preprocessor
    data_files = [
        data_directory / (variable_name + '.npy')
        for data_directory in data_directories]
    if preprocess_method is None:
        preprocessor = util.IdentityConverter()
    elif preprocess_method == 'standardize':
        preprocessor = util.Standardizer.lazy_read_files(data_files)
    elif preprocess_method == 'std_scale':
        preprocessor = util.StandardScaler.lazy_read_files(data_files)

    # Transform and save data
    for data_directory in data_directories:
        transformed_data = preprocessor.transform(
            util.load_variable(data_directory, variable_name))
        output_directory = determine_output_directory(
            data_directory, output_base_directory, str_replace)
        util.save_variable(output_directory, variable_name, transformed_data)
        (output_directory / finished_file).touch()

    return


def extract_variables(
        fem_data, mandatory_variables, *, optional_variables=None):
    """Extract variables from FEMData object to convert to data dictionary.

    Args:
        fem_data: femio.FEMData
            FEMData object to be extracted variables from.
        mandatory_variables: list of str
            Mandatory variable names.
        optional_variables: list of str, optional [None]
            Optional variable names.
    Returns:
        dict_data: dict
            Data dictionary.
    """
    dict_data = {
        mandatory_variable: fem_data.access_attribute(mandatory_variable)
        for mandatory_variable in mandatory_variables}
    for optional_variable in optional_variables:
        optional_variable_data = fem_data.access_attribute(
            optional_variable, mandatory=False)
        if optional_variable_data is not None:
            dict_data.update({optional_variable: optional_variable_data})
    return dict_data


def save_dict_data(output_directory, dict_data, *, dtype=np.float32):
    """Save dict_data.

    Args:
        output_directory: pathlib.Path
            Output directory path.
        dict_data: dict
            Data dictionary to be saved.
        dtype: type, optional [np.float32]
            Data type to be saved.
    Returns:
        None
    """
    for key, value in dict_data.items():
        util.save_variable(output_directory, key, value, dtype=dtype)
    return


def determine_output_directory(
        input_directory, output_base_directory, str_replace):
    """Determine output directory by replacing a string (str_replace) in the
    input_directory.

    Args:
        input_directory: pathlib.Path
            Input directory path.
        output_base_directory: pathlib.Path
            Output base directory path. The output directry name is under that
            directory.
        str_replace: str
            The string to be replaced.
    Output:
        output_directory: pathlib.Path
            Detemined output directory path.

    >>> determine_output_directory(
            Path('data/raw/a/b'), Path('test/sth'), 'raw')
    Path('test/sth/a/b')
    """
    replace_indices = np.where(
        np.array(input_directory.parts) == str_replace)[0]
    if len(replace_indices) != 1:
        raise ValueError(
            f"Input directory {input_directory} does not contain "
            f"{str_replace} directory or ambiguous.")

    replace_index = replace_indices[0]
    if replace_index + 1 == len(input_directory.parts):
        output_directory = output_base_directory
    else:
        output_directory = output_base_directory / '/'.join(
            input_directory.parts[replace_index + 1:])
    return output_directory


def normalize_adjacency_matrix(adj):
    """Symmetrically normalize adjacency matrix.

    Args:
        adj: scipy.sparse.coo_matrix
            Adjacency matrix in COO expression.
    Returns:
        normalized_adj: scipy.sparse.coo_matrix
            Normalized adjacency matrix in COO expression.
    """
    print(f"to_coo adj: {dt.datetime.now()}")
    adj = sp.coo_matrix(adj)
    print(f"sum raw: {dt.datetime.now()}")
    rowsum = np.array(adj.sum(1))
    print(f"invert d: {dt.datetime.now()}")
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    print(f"making diag: {dt.datetime.now()}")
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    print(f"calculating norm: {dt.datetime.now()}")
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

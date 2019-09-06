"""Module for preprocessing."""

import datetime as dt
import itertools as it
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from . import femio
from . import util
from . import setting


DTYPE = np.float32
FEMIO_FILE = 'femio_npy_saved.npy'


def convert_raw_data(
        raw_directory, mandatory_variables, *, optional_variables=None,
        output_base_directory='data/interim',
        recursive=False, conversion_function=None, force_renew=False,
        finished_file='converted', file_type='fistr',
        required_file_names=['*.msh', '*.cnt', '*.res.0.1'], read_npy=False,
        write_ucd=True):
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
        write_ucd: bool, optional [True]
            If True, write AVS UCD file with preprocessed variables.
    Returns:
        None
    """
    # Process all directories when raw directory is a list
    if isinstance(raw_directory, list) or isinstance(raw_directory, set):
        for _raw_directory in set(raw_directory):
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
            raw_directory, add_base=False)
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
        dict_data.update(conversion_function(fem_data, raw_directory))

    # Save data
    fem_data.save(output_directory)
    if write_ucd:
        write_ucd_file(output_directory, fem_data, dict_data)
    save_dict_data(output_directory, dict_data)
    (output_directory / finished_file).touch()

    return


def write_ucd_file(output_directory, fem_data, dict_data=None):
    if dict_data is not None:
        # Merge dict_data to fem_data
        for key, value in dict_data.items():
            if key in ['adj', 'nadj']:
                continue
            try:
                fem_data.elemental_data.update({
                    key: femio.FEMAttribute(
                        key, fem_data.elements.ids, value)})
            except ValueError:
                raise ValueError(key)

    fem_data.write('ucd', output_directory / 'mesh.inp')


class Preprocessor:

    REQUIRED_FILE_NAMES = ['converted']
    FINISHED_FILE = 'preprocessed'

    @classmethod
    def read_settings(cls, settings_yaml):
        preprocess_setting = setting.PreprocessSetting.read_settings_yaml(
            settings_yaml)
        return cls(preprocess_setting)

    def __init__(self, setting):
        self.setting = setting

    def preprocess_interim_data(self, *, force_renew=False):
        """Preprocess interim data with preprocessing e.g. standardization and then
        save them.

        Args:
            force_renew: bool, optional [False]
                If True, renew npy files even if they are alerady exist.
        Returns:
            None
        """
        interim_directories = util.collect_data_directories(
            self.setting.data.interim,
            required_file_names=self.REQUIRED_FILE_NAMES, add_base=False)
        if self.setting.data.pad:
            self.max_n_element = self._determine_max_n_element(
                interim_directories, list(self.setting.preprocess.keys())[0])

        # Preprocess data variable by variable
        dict_preprocessor_settings = {}
        for variable_name, preprocess_method \
                in self.setting.preprocess.items():
            parameters = self.preprocess_single_variable(
                interim_directories, variable_name, preprocess_method,
                str_replace='interim', force_renew=force_renew)
            dict_preprocessor_settings.update({
                variable_name:
                {'method': preprocess_method, 'parameters': parameters}})
        with open(
                self.setting.data.preprocessed / 'preprocessors.pkl',
                'wb') as f:
            pickle.dump(dict_preprocessor_settings, f)
        return

    def _determine_max_n_element(self, data_directories, variable_name):
        max_n_element = 0
        for data_directory in data_directories:
            data = util.load_variable(data_directory, variable_name)
            max_n_element = max(max_n_element, data.shape[0])
        return max_n_element

    def preprocess_single_variable(
            self, data_directories, variable_name, preprocess_method, *,
            str_replace='interim', force_renew=False):
        """Preprocess single variable.

        Args:
            data_directories: list of pathlib.Path
                Data directories.
            variable_name: str
                The name of the variable.
            preprocess_method: str
                Preprocess method name.
            str_replace: str, optional ['interim']
                Name to replace the input data base directory with.
            force_renew: bool, optional [False]
                If True, renew npy files even if they are alerady exist.
        Returns:
            preprocessor_parameters: dict
        """

        # Check if data already exists
        if not force_renew and np.any([
                util.files_exist(
                    determine_output_directory(
                        data_directory,
                        self.setting.data.preprocessed, str_replace),
                    [variable_name + '.*'])
                for data_directory in data_directories]):
            print(
                'Data already exists in '
                f"{self.setting.data.preprocessed}. Skipped.")
            exit()

        # Prepare preprocessor
        data_files = [
            data_directory / (variable_name + '.npy')
            for data_directory in data_directories]
        preprocessor = util.create_converter(
            preprocess_method, data_files=data_files)

        # Transform and save data
        for data_directory in data_directories:
            transformed_data = preprocessor.transform(
                util.load_variable(data_directory, variable_name))
            if self.setting.data.pad:
                transformed_data = util.pad_array(
                    transformed_data, self.max_n_element)

            output_directory = determine_output_directory(
                data_directory, self.setting.data.preprocessed, str_replace)
            util.save_variable(
                output_directory, variable_name, transformed_data)

            (output_directory / self.FINISHED_FILE).touch()

        yaml_file = self.setting.data.preprocessed / 'settings.yml'
        if not yaml_file.exists():
            setting.write_yaml(self.setting, yaml_file)
        return preprocessor.parameters


class Converter:

    def __init__(self, converter_parameters_pkl):
        self.converters = self._generate_converters(converter_parameters_pkl)

    def _generate_converters(self, converter_parameters_pkl):
        with open(converter_parameters_pkl, 'rb') as f:
            dict_preprocessor_settings = pickle.load(f)

        converters = {
            variable_name: util.generate_converter_from_dict(value)
            for variable_name, value in dict_preprocessor_settings.items()}
        return converters

    def preprocess(self, dict_data_x):
        converted_dict_data_x = {
            variable_name:
            self.converters[variable_name].transform(data)
            for variable_name, data in dict_data_x.items()}
        return converted_dict_data_x

    def postprocess(
            self, dict_data_x, dict_data_y, output_directory=None, *,
            save_x=False, write_simulation=False, write_npy=True,
            write_simulation_base=None, simulation_type='fistr',
            data_addition_function=None):
        """Postprocess data with inversely converting them.

        Args:
            dict_data_x: dict
                Dict of input data.
            dict_data_y: dict
                Dict of output data.
            output_directory: pathlib.Path, optional [None]
                Output directory path.
            save_x: bool, optional [False]
                If True, save input values in addition to output values.
            write_simulation: bool, optional [False]
                If True, write simulation data file(s) based on the inference.
            write_npy: bool, optional [True]
                If True, write npy files of inferences.
            write_simulation_base: pathlib.Path, optional [None]
                Base of simulation data to be used for write_simulation option.
                If not fed, try to find from the input directories.
            simulation_type: str, optional ['fistr']
                Simulation file type to write.
        Returns:
            inversed_dict_data_x: dict
                Inversed input data.
            inversed_dict_data_y: dict
                Inversed output data.
        """
        inversed_dict_data_x = {
            variable_name:
            self.converters[variable_name].inverse(data)
            for variable_name, data in dict_data_x.items()}
        inversed_dict_data_y = {
            variable_name:
            self.converters[variable_name].inverse(data)
            for variable_name, data in dict_data_y.items()}

        # Save data
        if output_directory is not None:
            if write_npy:
                if save_x:
                    self.save(inversed_dict_data_x, output_directory)
                self.save(inversed_dict_data_y, output_directory)
            if write_simulation:
                if write_simulation_base is None:
                    raise ValueError('No write_simulation_base fed.')
                self.write_simulation(
                    inversed_dict_data_y, output_directory,
                    write_simulation_base=write_simulation_base,
                    simulation_type=simulation_type,
                    data_addition_function=data_addition_function)

        return inversed_dict_data_x, inversed_dict_data_y

    def write_simulation(
            self, dict_data_y, output_directory, write_simulation_base, *,
            simulation_type='fistr', data_addition_function=None):
        fem_data = femio.FEMData.read_directory(
            simulation_type, write_simulation_base)
        for k, v in dict_data_y.items():
            fem_data.overwrite_attribute(k, v[0])
        if data_addition_function is not None:
            fem_data = data_addition_function(fem_data)

        if simulation_type == 'fistr':
            ext = ''
        elif simulation_type == 'ucd':
            ext = '.inp'
        fem_data.write(simulation_type, output_directory / ('mesh' + ext))
        return

    def save(self, data_dict, output_directory):
        if not output_directory.exists():
            output_directory.mkdir(parents=True)
        for variable_name, data in data_dict.items():
            np.save(output_directory / f"{variable_name}.npy", data)
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
        mandatory_variable: fem_data.convert_nodal2elemental(
            mandatory_variable, ravel=True)
        for mandatory_variable in mandatory_variables}
    if optional_variables is not None and len(optional_variables) > 0:
        for optional_variable in optional_variables:
            try:
                optional_variable_data = fem_data.convert_nodal2elemental(
                    optional_variable, ravel=True)
                dict_data.update({optional_variable: optional_variable_data})
            except ValueError:
                continue
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


def analyze_data_directories(
        data_directories, x_name, f_name, *, n_split=10, n_bin=100,
        out_directory=None, ref_index=0, plot=True):
    """Analyze data f_name with grid over x_name.

    Args:
        data_directories: List[pathlib.Path]
            List of data directories.
        x_name: str
            Name of x variable.
        f_name: str
            Name of f variable.
        n_split: int, optional, [10]
            The number to split x space.
        n_bin: int, optional, [100]
            The number of bins to draw histogram
        out_directory: pathlib.Path, optional, [None]
            Output directory path. By default no output is written.
        ref_index: int, optional, [0]
            Reference data directory index to analyze data.
        plot: bool, optional, [True]
            If True, plot data by grid.
    """

    # Initialization
    if out_directory is not None:
        out_directory.mkdir(parents=True, exist_ok=True)

    data = [
        _read_analyzing_data(data_directory, x_name, f_name)
        for data_directory in data_directories]
    xs = [d[0] for d in data]
    fs = [d[1] for d in data]
    f_grids = _generate_grids(fs, n_bin, symmetric=True, magnitude_range=.1)

    ranges, list_split_data, centers, means, stds = split_data_arrays(
        xs, fs, n_split=n_split)

    # Plot data
    if plot:
        for range_, split_data in zip(ranges, list_split_data):
            range_string = '__'.join(f"{r[0]:.3e}_{r[1]:.3e}" for r in range_)
            if out_directory is None:
                out_file_base = None
            else:
                out_file_base = out_directory / f"{f_name}_{range_string}"
            _plot_histogram(
                split_data, f_grids, data_directories,
                out_file_base=out_file_base)

    # Write output file
    array_means = np.transpose(np.stack(means), (1, 0, 2))
    mean_diffs = array_means - array_means[ref_index]
    array_stds = np.transpose(np.stack(stds), (1, 0, 2))
    std_diffs = array_stds - array_stds[ref_index]

    header = ','.join(
        f"{x_name}_{i}" for i in range(centers.shape[-1])) + ',' \
        + ','.join(
            f"mean_diff_{f_name}_{i}" for i in range(mean_diffs.shape[-1])) \
        + ',mean_diff_norm,' \
        + ','.join(
            f"std_diff_{f_name}_{i}" for i in range(mean_diffs.shape[-1])) \
        + ',std_diff_norm'
    for i_dir, data_directory in enumerate(data_directories):
        mean_diff_norms = np.linalg.norm(mean_diffs[i_dir], axis=1)[:, None]
        std_diff_norms = np.linalg.norm(std_diffs[i_dir], axis=1)[:, None]
        np.savetxt(
            out_directory / (data_directory.stem + '.csv'),
            np.concatenate(
                [centers, mean_diffs[i_dir], mean_diff_norms,
                 std_diffs[i_dir], std_diff_norms], axis=1),
            delimiter=',', header=header)


def split_data_arrays(xs, fs, *, n_split=10):
    """Split data fs with regards to grids of xs.

    Args:
        xs: List[numpy.ndarray]
            n_sample-length list contains (n_element, dim_x) shaped ndarray.
        fs: List[numpy.ndarray]
            n_sample-length list contains (n_element, dim_f) shaped ndarray.
        n_split: int, optional, [10]
            The number to split x space.
    """

    x_grids = _generate_grids(xs, n_split)
    # raise ValueError(x_grids)

    # Analyze data by grid
    ranges = np.transpose(
        np.stack([x_grids[:-1, :], x_grids[1:, :]]), (2, 1, 0))
    useful_ranges = []
    list_split_data = []
    centers = []
    means = []
    stds = []
    for rs in it.product(*ranges):
        filters = [_calculate_filter(x, rs) for x in xs]
        if np.any([np.all(~filter_) for filter_ in filters]):
            continue

        filtered_fs = [f_[filter_] for f_, filter_ in zip(fs, filters)]
        list_split_data.append(filtered_fs)
        useful_ranges.append(rs)

        filtered_means = np.stack([np.mean(ff, axis=0) for ff in filtered_fs])
        filtered_stds = np.stack([np.std(ff, axis=0) for ff in filtered_fs])
        center = [np.mean(r) for r in rs]
        centers.append(center)
        means.append(filtered_means)
        stds.append(filtered_stds)

    # Write output file
    centers = np.array(centers)

    return useful_ranges, list_split_data, centers, means, stds


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
            plt.savefig(str(out_file_base) + f"_{i_dim}.png")
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
    return np.stack([
        [np.min(concat_data[:, i]), np.max(concat_data[:, i])]
        for i in range(concat_data.shape[-1])], axis=1)


def _read_analyzing_data(data_directory, x, f):
    fem_data = femio.FEMData.read_directory('fistr', data_directory)
    x_val = fem_data.convert_nodal2elemental(x, calc_average=True)
    f_val = fem_data.convert_nodal2elemental(f, calc_average=True)
    return x_val, f_val

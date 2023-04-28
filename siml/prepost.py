"""Module for preprocessing."""

import datetime as dt
import itertools as it
from typing import Union

import femio
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from siml import util
from siml.utils import fem_data_utils
from siml.preprocessing import ScalersComposition
from siml.siml_variables import ArrayDataType


def add_difference(
        fem_data, dict_data, reference_dict_data, prefix='difference'):
    if reference_dict_data is None:
        return fem_data
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

    wrapped_data = fem_data_utils.FemDataWrapper(fem_data)
    wrapped_data.update_fem_data(
        difference_dict_data, prefix=prefix
    )

    return wrapped_data.fem_data


def add_abs_difference(
        fem_data, dict_data, reference_dict_data, prefix='difference_abs'):
    if reference_dict_data is None:
        return fem_data
    intersections = set(
        dict_data.keys()).intersection(reference_dict_data.keys())
    if len(intersections) == 0:
        return fem_data

    difference_dict_data = {
        intersection:
        np.abs(
            np.reshape(
                dict_data[intersection],
                reference_dict_data[intersection].shape)
            - reference_dict_data[intersection])
        for intersection in intersections}

    wrapped_data = fem_data_utils.FemDataWrapper(fem_data)
    wrapped_data.update_fem_data(
        difference_dict_data,
        prefix=prefix
    )

    return wrapped_data.fem_data


def concatenate_preprocessed_data(
        preprocessed_base_directories, output_directory_base, variable_names,
        *, ratios=(.9, .05, .05), overwrite=False,
        finished_file='preprocessed'):
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
        required_file_names=finished_file)
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


class Converter:

    def __init__(self, converter_parameters_pkl, key=None):
        self.converters = ScalersComposition.create_from_file(
            converter_parameters_pkl=converter_parameters_pkl,
            key=key
        )

    def preprocess(self, dict_data_x: dict):
        input_dict = {
            name: v for name, v in dict_data_x.items()
            if name in self.converters.get_variable_names()
        }

        converted_dict_data_x = self.converters.transform_dict(input_dict)
        if len(converted_dict_data_x) == 0:
            raise ValueError(
                'No converted data found. '
                'Check the preprocessed directory set correctly.')
        return converted_dict_data_x

    def postprocess(
            self, dict_data_x: dict, dict_data_y: dict, output_directory=None,
            *,
            dict_data_y_answer=None,
            overwrite=False, save_x=False, write_simulation=False,
            write_npy=True, write_simulation_stem=None,
            write_simulation_base=None, read_simulation_type='fistr',
            save_function=None,
            write_simulation_type='fistr', skip_femio=False,
            load_function=None, convert_to_order1=False,
            data_addition_function=None, required_file_names=[],
            less_output=False, perform_inverse=True,
            skip_fem_data_creation=False, **kwargs):
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
        less_output: bool, optional
            If True, output less variables in FEMData object.
        data_addition_function: callable, optional
            Function to add some data to existing fem_data.
        skip_fem_data_creation: bool, optional
            If True, skip fem_data object creation.

        Returns
        --------
            inversed_dict_data_x: dict
                Inversed input data.
            inversed_dict_data_y: dict
                Inversed output data.
            inversed_dict_answer: dict
                Inversed answer data. None when the answer not available.
            fem_data: femio.FEMData
                FEMData object with input and output data.
        """
        _dict_data_x = self._format_dict_shape(dict_data_x)
        _dict_data_y = self._format_dict_shape(dict_data_y)
        _dict_data_y_answer = self._format_dict_shape(dict_data_y_answer)

        if perform_inverse:
            _dict_data_x = self._inverse_process(_dict_data_x)
            _dict_data_y = self._inverse_process(_dict_data_y)
            _dict_data_y_answer = self._inverse_process(_dict_data_y_answer)

        # Save data
        if skip_fem_data_creation or write_simulation_base is None \
                or not write_simulation_base.exists():
            fem_data = None
        else:
            try:
                fem_data = self._create_fem_data(
                    _dict_data_x, _dict_data_y,
                    dict_data_answer=_dict_data_y_answer,
                    write_simulation_base=write_simulation_base,
                    write_simulation_stem=write_simulation_stem,
                    read_simulation_type=read_simulation_type,
                    data_addition_function=data_addition_function,
                    skip_femio=skip_femio, load_function=load_function,
                    convert_to_order1=convert_to_order1,
                    required_file_names=required_file_names)
            except ValueError as e:
                print(
                    f"{e}\n"
                    'Could not read FEMData object, set None\n'
                    f"write_simulation_base: {write_simulation_base}\n"
                    f"write_simulation_stem: {write_simulation_stem}\n"
                    f"read_simulation_type: {read_simulation_type}\n"
                )
                fem_data = None
                write_simulation_base = None
                write_simulation = False
        if output_directory is not None:
            if write_npy:
                if save_x:
                    self.save(_dict_data_x, output_directory)
                self.save(_dict_data_y, output_directory)
            if write_simulation:
                if write_simulation_base is None:
                    raise ValueError('No write_simulation_base fed.')
                self._write_simulation(
                    output_directory, fem_data, overwrite=overwrite,
                    write_simulation_type=write_simulation_type,
                    less_output=less_output)
            if save_function is not None:
                save_function(
                    output_directory, fem_data, overwrite=overwrite,
                    write_simulation_type=write_simulation_type)

        return _dict_data_x, _dict_data_y, \
            _dict_data_y_answer, fem_data

    def _inverse_process(
        self,
        dict_data: Union[dict[str, ArrayDataType], None]
    ) -> dict[str, ArrayDataType]:
        if dict_data is None:
            return {}

        dict_data_answer = self.converters.inverse_transform_dict(dict_data)
        return dict_data_answer

    def _format_dict_shape(
        self,
        dict_data: Union[dict, None]
    ) -> Union[dict[str, ArrayDataType], None]:
        # This function should be deprecated
        # It is not appropriate to overwrite value for variable name
        if dict_data is None:
            return None

        if len(dict_data) == 0:
            return None

        if isinstance(list(dict_data.values())[0], dict):
            return_dict_data = {
                variable_name: data
                for value in dict_data.values()
                for variable_name, data in value.items()
                if variable_name in self.converters.get_variable_names()
            }
        else:
            return_dict_data = {
                variable_name: data
                for variable_name, data in dict_data.items()
                if variable_name in self.converters.get_variable_names()
            }
        return return_dict_data

    def _create_fem_data(
            self, dict_data_x, dict_data_y, write_simulation_base, *,
            dict_data_answer=None,
            write_simulation_stem=None,
            read_simulation_type='fistr', data_addition_function=None,
            skip_femio=False, load_function=None,
            required_file_names=[], convert_to_order1=False):
        if not skip_femio:
            fem_data = femio.FEMData.read_directory(
                read_simulation_type, write_simulation_base,
                stem=write_simulation_stem, save=False,
                read_mesh_only=False)
        elif load_function:
            if len(required_file_names) == 0:
                raise ValueError(
                    'Please specify required_file_names when skip_femio '
                    'is True.')
            data_files = util.collect_files(
                write_simulation_base, required_file_names)
            data_dict, fem_data = load_function(
                data_files, write_simulation_base)
            wrapped_fem_data = fem_data_utils.FemDataWrapper(fem_data)
            wrapped_fem_data.update_fem_data(data_dict, allow_overwrite=True)
            fem_data = wrapped_fem_data.fem_data
        else:
            raise ValueError(
                'When skip_femio is True, please feed load_function.')

        if convert_to_order1:
            fem_data = fem_data.to_first_order()

        wrapped_fem_data = fem_data_utils.FemDataWrapper(fem_data)
        wrapped_fem_data.update_fem_data(dict_data_x, prefix='input_')
        if dict_data_answer is not None:
            wrapped_fem_data.update_fem_data(
                dict_data_answer, prefix='answer_'
            )
        wrapped_fem_data.update_fem_data(dict_data_y, prefix='predicted_')

        fem_data = wrapped_fem_data.fem_data
        fem_data = add_difference(
            fem_data, dict_data_y, dict_data_answer, prefix='difference_')
        fem_data = add_abs_difference(
            fem_data, dict_data_y, dict_data_answer, prefix='difference_abs_')
        if data_addition_function is not None:
            fem_data = data_addition_function(fem_data, write_simulation_base)

        return fem_data

    def _write_simulation(
            self, output_directory, fem_data, *,
            write_simulation_type='fistr', overwrite=False, less_output=False):
        if write_simulation_type == 'fistr':
            ext = ''
        elif write_simulation_type == 'ucd':
            ext = '.inp'
        elif write_simulation_type == 'vtk':
            ext = '.vtk'
        elif write_simulation_type in ['polyvtk', 'vtu']:
            ext = '.vtu'
        else:
            raise ValueError(
                f"Unexpected write_simulation_type: {write_simulation_type}")
        if less_output:
            nodal_data = {}
            for key in fem_data.nodal_data.keys():
                if 'answer_' in key or 'predicted_' in key \
                        or 'difference_' in key:
                    nodal_data.update({
                        key: fem_data.nodal_data.get_attribute_data(key)})
                else:
                    pass
            fem_data.nodal_data.reset()

            elemental_data = {}
            for key in fem_data.elemental_data.keys():
                if 'answer_' in key or 'predicted_' in key \
                        or 'difference_' in key:
                    elemental_data.update({
                        key: fem_data.elemental_data.get_attribute_data(key)})
                else:
                    pass
            if 'face' in fem_data.elemental_data:
                face = fem_data.elemental_data['face']
                has_face = True
            else:
                has_face = False
            fem_data.elemental_data.reset()
            fem_data.nodal_data.update_data(fem_data.nodes.ids, nodal_data)
            fem_data.elemental_data.update_data(
                fem_data.elements.ids, elemental_data)
            if has_face:
                fem_data.elemental_data['face'] = face

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

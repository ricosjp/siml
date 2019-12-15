import datetime as dt
from glob import glob
import io
import itertools as it
import os
from pathlib import Path
import subprocess

import chainer as ch
import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn import preprocessing
import yaml

from .femio import FEMData, FEMAttribute


INFERENCE_FLAG_FILE = 'inference'
SPARSE_DATA_NAMES = ['adj', 'nadj']


def date_string():
    return dt.datetime.now().isoformat().replace('T', '_').replace(':', '-')


def load_yaml_file(file_name):
    """Load YAML file.

    Args:
        file_name: str or pathlib.Path
            YAML file name.
    Returns:
        dict_data: dict
            YAML contents.
    """
    with open(file_name, 'r') as f:
        dict_data = yaml.load(f, Loader=yaml.SafeLoader)
    return dict_data


def load_yaml(source):
    """Load YAML source.

    Args:
        source: File-like object or str or pathlib.Path
    Returns:
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
        output_directory, file_basename, data, *, dtype=np.float32):
    """Save variable data.

    Args:
        output_directory: pathlib.Path
            Save directory path.
        file_basename: str
            Save file base name without extenstion.
        data: np.ndarray or scipy.sparse.coo_matrix
            Data to be saved.
        dtype: type, optional [np.float32]
            Data type to be saved.
    Returns:
        None
    """
    if not output_directory.exists():
        output_directory.mkdir(parents=True)
    if isinstance(data, np.ndarray):
        save_file_path = output_directory / (file_basename + '.npy')
        np.save(
            output_directory / (file_basename + '.npy'), data.astype(dtype))
    elif isinstance(data, sp.coo_matrix):
        save_file_path = output_directory / (file_basename + '.npz')
        sp.save_npz(save_file_path, data.astype(dtype))
    else:
        raise ValueError(f"{file_basename} has unknown type: {data.__class__}")

    print(f"{file_basename} is saved in: {save_file_path}")
    return


def load_variable(data_directory, file_basename):
    """Load variable data.

    Args:
        output_directory: pathlib.Path
            Directory path.
        file_basename: str
            File base name without extenstion.
    Returns:
        data: numpy.ndarray or scipy.sparse.coo_matrix
    """
    if file_basename in SPARSE_DATA_NAMES:
        return sp.load_npz(data_directory / (file_basename + '.npz'))
    else:
        return np.load(data_directory / (file_basename + '.npy'))


def collect_data_directories(
        base_directory, *, required_file_names=None):
    """Collect data directories recursively from the base directory.

    Args:
        base_directory: pathlib.Path
            Base directory to search directory from.
        required_file_names: list of str
            If given, only return directories which have required files.
    Returns:
        found_directories: list of pathlib.Path
            All found directories.
    """
    if not base_directory.exists():
        raise ValueError(f"{base_directory} not exist")

    if required_file_names:
        found_directories = [
            Path(directory) for directory, _, sub_files
            in os.walk(base_directory, followlinks=True)
            if len(sub_files) > 0 and files_match(
                    sub_files, required_file_names)]
    else:
        found_directories = [
            Path(directory) for directory, _, sub_files
            in os.walk(base_directory, followlinks=True)]
    return found_directories


def collect_files(directory, required_file_names):
    """Collect data directories recursively from the base directory.

    Args:
        base_directory: pathlib.Path
            Base directory to search directory from.
        required_file_names: list of str
    """
    return [
        found_file
        for required_file_name in required_file_names
        for found_file in glob(str(directory / required_file_name))
    ]


def files_match(file_names, required_file_names):
    """Check if file names match.

    Args:
        file_names: List[str]
        file_names: List[str]
    Returns:
        files_match: bool
            True if all files match. Otherwise False.
    """
    return np.all([
        np.any([
            required_file_name.lstrip('*') in file_name
            for file_name in file_names])
        for required_file_name in required_file_names])


def files_exist(directory, file_names):
    """Check if files exist in the specified directory.

    Args:
        directory: pathlib.Path
        file_names: list of str
    Returns:
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

    def __init__(self, setting_data, *, data_files=None, componentwise=True):
        if isinstance(setting_data, dict):
            self._init_with_dict(setting_data)
        elif isinstance(setting_data, str):
            self._init_with_str(setting_data)
        elif isinstance(setting_data, BaseEstimator):
            self._init_with_converter(setting_data)
        else:
            raise ValueError(f"Unsupported setting_data: {setting_data}")

        if data_files is not None:
            self.lazy_read_files(data_files, componentwise)
        return

    def _init_with_dict(self, setting_dict):
        preprocess_method = setting_dict['method']
        self._init_with_str(preprocess_method)
        return

    def _init_with_str(self, preprocess_method):
        if preprocess_method == 'identity':
            self.converter = Identity()
        elif preprocess_method == 'standardize':
            self.converter = preprocessing.StandardScaler()
        elif preprocess_method == 'std_scale':
            self.converter = preprocessing.StandardScaler(with_mean=False)
        elif preprocess_method == 'min_max':
            self.converter = preprocessing.MinMaxScaler()
        elif preprocess_method == 'min_abs':
            self.converter = preprocessing.MaxAbsScaler()
        else:
            raise ValueError(
                f"Unknown preprocessing method: {preprocess_method}")
        return

    def _init_with_converter(self, converter):
        self.converter = converter
        return

    def lazy_read_files(self, data_files, componentwise=True):
        for data_file in data_files:
            data = np.load(data_file)
            if not componentwise:
                data = np.reshape(data, (-1, 1))
            self.converter.partial_fit(data)
        return

    def transform(self, data):
        return self.converter.transform(data)

    def inverse(self, data):
        return self.converter.inverse_transform(data)


class Identity(TransformerMixin, BaseEstimator):
    """Class to perform identity conversion (do nothing)."""

    def partial_fit(cls, data):
        return

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


def diagonalize(data, rotations):
    matrices = np.array(
        [r @ array2symmat(d) @ r.T for d, r in zip(data, rotations)])
    # print(np.max([m[~np.eye(3, dtype=bool)] for m in matrices]))
    return extract_diag(matrices)


def anti_diagonalize(data, rotations):
    return np.array([r.T @ np.diag(d) @ r for d, r in zip(data, rotations)])


def symmat2array(symmat, to_engineering=False):
    """Convert symmetric matrix to array with 6 components."""
    if len(symmat.shape) == 2:  # One matrix
        arr = _single_symmat2array(symmat)
    elif len(symmat.shape) == 3:  # List of matrices
        arr = np.array([_single_symmat2array(m) for m in symmat])
    else:
        raise ValueError
    if to_engineering:
        arr[:, 3:] = arr[:, 3:] * 2
    return arr


def _single_symmat2array(symmat):
    try:
        assert abs(symmat[0, 1] - symmat[1, 0]) < 1e-5
        assert abs(symmat[0, 2] - symmat[2, 0]) < 1e-5
        assert abs(symmat[1, 2] - symmat[2, 1]) < 1e-5
    except AssertionError:
        raise ValueError(symmat)

    return np.array(
        [symmat[0, 0], symmat[1, 1], symmat[2, 2],
         symmat[0, 1], symmat[1, 2], symmat[0, 2]])


def array2symmat(array, from_engineering=False):
    """Convert array with 6 components to symmetric matrix."""
    if len(array.shape) == 1:  # Single array
        arr = _single_array2symmat(array)
    elif len(array.shape) == 2:  # List of h
        arr = np.array([_single_array2symmat(a) for a in array])
    else:
        raise ValueError
    if from_engineering:
        arr[:, 0, 1] = arr[:, 0, 1] / 2
        arr[:, 0, 2] = arr[:, 0, 2] / 2
        arr[:, 1, 2] = arr[:, 1, 2] / 2
        arr[:, 1, 0] = arr[:, 1, 0] / 2
        arr[:, 2, 0] = arr[:, 2, 0] / 2
        arr[:, 2, 1] = arr[:, 2, 1] / 2
    return arr


def _single_array2symmat(array):
    a = array
    return np.array([
        [a[0], a[3], a[5]],
        [a[3], a[1], a[4]],
        [a[5], a[4], a[2]]
    ])


def extract_diag(mat):
    if len(mat.shape) == 2:  # Single matrix
        return _extract_single_diag(mat)
    elif len(mat.shape) == 3:  # List of matrices
        return np.array([_extract_single_diag(m) for m in mat])


def _extract_single_diag(mat):
    return np.array([mat[0, 0], mat[1, 1], mat[2, 2]])


def calculate_ansys_angles(orientations):
    # Just inverse ansys -> frontistr
    x_rad = np.arcsin(orientations[:, 5])

    # Use arctan2 to handle pi / 2 * n case
    z_rad = np.arctan2(- orientations[:, 3], orientations[:, 4])

    # Use arccos to have range [0, pi]
    b = orientations[:, 0] * np.cos(z_rad) + orientations[:, 1] * np.sin(z_rad)
    b[b > 1.] = 1.
    b[b < -1.] = -1.
    y_rad = - np.arccos(b) * np.sign(orientations[:, 2])

    return np.stack([z_rad, x_rad, y_rad], axis=1) / np.pi * 180


def calculate_rotation_angles(orientations, *, standardize=False):
    """Calculate rotation angles w.r.t global axes.

    Args:
        orients: 2-D orientation data in FrontISTR style.
        standardize: Convert range of outputs to [-.5, .5].
    Returns:
        [[theta_x, theta_y, theta_z], ...], where each theta is corresponding
        to the rotation angle w.r.t each exis (Euler angles).
    """
    rotations = generate_rotation_matrices(
        orientations[:, :3], orientations[:, 3:6])
    thetas_x = [np.arctan2(r[2, 1], r[2, 2]) for r in rotations]
    thetas_y = [np.arctan2(-r[2, 0], (r[2, 1]**2 + r[2, 2]**2)**0.5)
                for r in rotations]
    thetas_z = [np.arctan2(r[1, 0], r[0, 0]) for r in rotations]

    return np.array([thetas_x, thetas_y, thetas_z]).T


def calculate_natural_element_shape(fem_data):
    """Calculate element shape in the natural coordinate. The shape is
    expressed in the relative position viewed from the first node, in the
    natural coordinate, where 1st axis is 1-2 vector, 1-2 plain is
    span(1-2 vector, 1-3 vector).

    Args:
        fem_data: FEMData object
    Returns:
        nshape: numpy.ndarray
            [n_node, m] shaped ndarray,
            where m = (order1_n_node_per_element - 1) * 3 - 3.
            -1 because the first node is always at [0, 0, 0],
            -3 because the second node is always at [r, 0, 0] and the third
            node is always at [s_1, s_2, 0] so ommit components which are
            always zero.
    """
    n_node_per_element = fem_data.elements.data.shape[1]
    if n_node_per_element == 4:
        n_node = 4
    elif n_node_per_element == 10:
        n_node = 4
    else:
        raise ValueError(
            f"Unsupported # of nodes per element: {n_node_per_element}")

    # Assume element type is tetrahedron
    node_positions = np.array([
        fem_data.nodes.data[fem_data.nodes.ids2indices(
            nodes, fem_data.dict_node_id2index), :]
        for nodes in fem_data.elements.data[:, :n_node]])
    node_relative_positions = np.reshape(
        node_positions[:, 1:, :] - node_positions[:, 0, None, :],
        (len(node_positions), -1))

    pos1 = np.linalg.norm(node_relative_positions[:, :3], axis=1)
    axis1 = (node_relative_positions[:, :3].T / pos1).T
    _axis2 = node_relative_positions[:, 3:6]
    axis3 = _normalize(np.cross(axis1, _axis2))
    axis2 = np.cross(axis3, axis1)

    pos2 = np.stack([
        np.einsum('ij,ij->i', axis1, node_relative_positions[:, 3:6]),
        np.einsum('ij,ij->i', axis2, node_relative_positions[:, 3:6])]).T
    pos3 = np.stack([
        np.einsum('ij,ij->i', axis1, node_relative_positions[:, 6:]),
        np.einsum('ij,ij->i', axis2, node_relative_positions[:, 6:]),
        np.einsum('ij,ij->i', axis3, node_relative_positions[:, 6:])]).T

    nshape = np.concatenate([
        pos1[:, None], pos2, pos3], axis=1)
    return nshape


def calculate_element_position(fem_data):
    """Calculate position of element.

    Args:
        fem_data: FEMData object
    Returns:
        averaged_element_positions: numpy.ndarray
            [n_element, 3] shaped array indicating the centor of mass of
            each element.
        element_positions: numpy.ndarray
            [n_element, 3 * order1_node_per_element] shaped array indicating
            node positions associated each element.
    """
    n_node_per_element = fem_data.elements.data.shape[1]
    if n_node_per_element == 4:
        n_node = 4
    elif n_node_per_element == 10:
        n_node = 4
    else:
        raise ValueError(
            f"Unsupported # of nodes per element: {n_node_per_element}")

    # Assume element type is tetrahedron
    node_positions = np.array([
        fem_data.nodes.data[fem_data.nodes.ids2indices(
            nodes, fem_data.dict_node_id2index), :]
        for nodes in fem_data.elements.data[:, :n_node]])
    element_positions = np.reshape(node_positions, (-1, 12))
    averaged_element_positions = np.stack([
        np.mean(element_positions[:, 0::3], axis=1),
        np.mean(element_positions[:, 1::3], axis=1),
        np.mean(element_positions[:, 2::3], axis=1),
    ], axis=1)
    return averaged_element_positions, element_positions


def calculate_adjacency_matrix(fem_data, *, n_node=None):
    """Calculate graph adjacency matrix regarding elements sharing the same
    node as connected.

    Args:
        fem_data: FEMData object
        n_node: int, optional [None]
            the number of node of interest. n_node = 4 to extract only order
            1 nodes in tet2 mesh.
    Returns:
        adj: scipy.sparse.coo_matrix
            Adjacency matrix in COO expression.
    """
    if n_node is None:
        n_node = fem_data.elements.data.shape[1]
    print('Calculating map from node to elements')
    print(dt.datetime.now())
    nodeid2elemid = fem_data.calculate_dict_node_id_to_element_id()
    print('Calculating map from element to elements')
    print(dt.datetime.now())
    element2elements = {
        e: np.unique(np.concatenate([
            nodeid2elemid[d] for d in data]))
        for e, data in zip(
                fem_data.elements.ids, fem_data.elements.data[:, :n_node])}
    print('Creating graph')
    print(dt.datetime.now())
    graph = nx.from_dict_of_lists(element2elements)
    print('Creating adj')
    print(dt.datetime.now())
    return nx.adjacency_matrix(graph).tocoo()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix.

    Args:
        adj: scipy.sparse.coo_matrix
            Adjacency matrix in COO expression.
    Returns:
        normalized_adj: scipy.sparse.coo_matrix
            Normalized adjacency matrix in COO expression.
    """
    print('to_coo adj')
    print(dt.datetime.now())
    adj = sp.coo_matrix(adj)
    print('sum raw')
    print(dt.datetime.now())
    rowsum = np.array(adj.sum(1))
    print('invert d')
    print(dt.datetime.now())
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    print('making diag')
    print(dt.datetime.now())
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    print('calculating norm')
    print(dt.datetime.now())
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def calculate_mesh_shape(fem_data, *, n_node=None):
    """Calculate mesh shape data.

    Args:
        fem_data: FEMData objects.
        n_node: The number of node to consider (default: use all nodes).
    Returns:
        [[d_12_x, d_12_y, d_12_z, d_13_x, ..., d_23_x, ...], ...], where each d
        is corresponding to the distance between node_i and node_j in one
        element. It with be [n_element, 3 * (n_node_per_element C 2)] shaped
        array.
    """
    if n_node is None:
        n_node = fem_data.elements.data.shape[1]
    node_positions = [
        fem_data.nodes.data[fem_data.nodes.ids2indices(
            nodes, fem_data.dict_node_id2index), :]
        for nodes in fem_data.elements.data[:, :n_node]]
    shape = np.concatenate(
        [c[0] - c[1]
         for c in it.combinations(np.transpose(node_positions, (1, 0, 2)), 2)],
        axis=1)
    return shape


def calculate_node_position(fem_data, *, n_node=None):
    """Calculate node relative positions.

    Args:
        fem_data: FEMData objects.
        n_node: The number of node to consider (default: use all nodes).
    Returns:
        [[d_12_x, d_12_y, d_12_z, d_13_x, ...], ...], where each d is
        corresponding to the distance between node_i and node_j in one element.
        It with be [n_element, 3 * (n_node_per_element C 2)] shaped array.
    """
    if n_node is None:
        n_node = fem_data.elements.data.shape[1]
    node_positions = np.array([
        fem_data.nodes.data[fem_data.nodes.ids2indices(
            nodes, fem_data.dict_node_id2index), :]
        for nodes in fem_data.elements.data[:, :n_node]])
    node_relative_positions = np.reshape(
        node_positions[:, 1:, :] - node_positions[:, 0, None, :],
        (len(node_positions), -1))
    return node_relative_positions


def _normalize(xs):
    if len(xs.shape) != 2:
        raise ValueError
    return (xs.T / np.linalg.norm(xs, axis=1)).T


def generate_rotation_matrices(xs, ys):
    normal_xs = _normalize(xs)
    normal_ys = _normalize(ys)
    normal_zs = _normalize(np.cross(normal_xs, normal_ys))
    ortho_normal_ys = np.cross(normal_zs, normal_xs)
    return np.array([np.array([x, y, z]).T for x, y, z
                     in zip(normal_xs, ortho_normal_ys, normal_zs)])


def collect_variable(list_fem_data, variable_name, *,
                     is_elemental=True):
    if is_elemental:
        return np.concatenate(
            [fem_data.elemental_data[variable_name].data
             for fem_data in list_fem_data])
    else:
        return np.concatenate(
            [fem_data.convert_nodal2elemental(variable_name, ravel=True)
             for fem_data in list_fem_data])


def extract_variable(fem_data, variable_name, *, is_elemental=True):
    if is_elemental:
        return fem_data.elemental_data[variable_name].data
    else:
        return fem_data.convert_nodal2elemental(variable_name, ravel=True)


def save_data(dir_name, base_name, data):
    path_name = os.path.join(dir_name, base_name + '.npy')
    np.save(path_name, data)
    print('Save {} in: {}'.format(base_name, path_name))


def save_npz(dir_name, base_name, data):
    path_name = os.path.join(dir_name, base_name + '.npz')
    sp.save_npz(path_name, data)
    print('Save {} in: {}'.format(base_name, path_name))


def load_npz(dir_name, base_name):
    path_name = os.path.join(dir_name, base_name + '.npz')
    return sp.load_npz(path_name)


def concat_dicts(dicts):
    """Contatinate list of dicts."""
    concated_dic = {}
    for d in dicts:
        concated_dic.update(d)
    return concated_dic


def dir2name(dir_name):
    if isinstance(dir_name, list):
        return '_'.join([dir2name(_) for _ in dir_name])
    name = dir_name.replace('/', '_')
    if name[-1] == '_':
        return name[:-1]
    else:
        return name


def concat_examples(batch, device=None):
    x = ch.dataset.to_device(
        device, ch.functions.pad_sequence([b['x'] for b in batch]).data)
    original_lengths = [b['x'].shape[0] for b in batch]
    t = ch.dataset.to_device(
        device, np.concatenate([b['t'] for b in batch]))
    return {'x': x, 't': t, 'original_lengths': original_lengths}


def generate_converter(support_train):
    # NOTE: Assume orders are the same for all support inputs
    sparse = support_train[0][0]
    order = ch.utils.get_order(sparse.row, sparse.col)

    def convert_example_with_support(batch, device=None):
        x = ch.dataset.to_device(
            device, ch.functions.pad_sequence([b['x'] for b in batch]).data)
        original_lengths = [b['x'].shape[0] for b in batch]
        t = ch.dataset.to_device(
            device, np.concatenate([b['t'] for b in batch]))
        length = x.shape[1]
        supports = [
            [
                convert_sparse_to_chainer(
                    pad_sparse(sparse, length), device=device, order=order)
                for sparse in b['supports']]
            for b in batch]
        return {
            'x': x, 't': t, 'supports': supports,
            'original_lengths': original_lengths}

    return convert_example_with_support


def pad_sparse_sequence(sparse_sequence):
    length = np.max([s.shape[0] for s in sparse_sequence])
    return [pad_sparse(s, length) for s in sparse_sequence]


def pad_sparse(sparse_matrix, length):
    return sp.coo_matrix(
        (sparse_matrix.data, (sparse_matrix.row, sparse_matrix.col)),
        shape=(length, length))


def element_wise_converter(batch, device=None):
    x = ch.dataset.to_device(
        device, np.concatenate([b['x'] for b in batch]))
    t = ch.dataset.to_device(
        device, np.concatenate([b['t'] for b in batch]))
    return {'x': x, 't': t}


def convert_sparse_to_chainer(_sparse, *, device=-1, order=None, xp=np):
    sparse = ch.utils.CooMatrix(
        ch.dataset.to_device(device, _sparse.data),
        ch.dataset.to_device(device, _sparse.row),
        ch.dataset.to_device(device, _sparse.col),
        _sparse.shape, order='C')
    return sparse


def calc_eigs_symmetric_sparse(sparse_mat):
    print('eig')
    dim = sparse_mat.shape[0]
    k1 = dim // 2
    k2 = dim - k1
    w1, v1 = sp.linalg.eigsh(sparse_mat, k=k1, which='LM')
    w2, v2 = sp.linalg.eigsh(sparse_mat, k=k2, which='SM')
    print(w1, w2)


def align_data(points):
    """Make alignments of point cloud by doing mean subtraction and PCA.
    If you make transformation by yourself, you have to perform something
    equivalent to (points - means) @ v.T

    Args:
        points: [n_data, dim] ndarray.
    Return
        rotated_points: [n_data, dim] ndarray which is result of the alignment.
        v: [dim, dim] ndarray which is rotation matrix. You have to perform
            points @ v.T to make correct transformation.
        means: [dim] ndarray of means.
    """
    print('Doing PCA')
    means = np.mean(points, axis=0)
    centered_points = points - means

    pca = PCA(n_components=3)
    pca.fit(points)
    v = pca.components_

    # Sort eigenvectors with eigenvalues
    # cov = points.T @ points / (points.shape[0] - 1)
    # W, V_pca = np.linalg.eigh(cov)
    # index = W.argsort()[::-1]
    # W = W[index]
    # v = V_pca[:, index]

    # v[:, 2] = np.cross(v[:, 0], v[:, 1])
    v[2, :] = np.cross(v[0, :], v[1, :])

    # # Manage orthogonal matrix's det < 0 case by swapping axes
    # if np.linalg.det(v) < 0.:
    #     print('Axes swapped because det of the orthogonal matrix < 0')
    #     v = np.stack([v[:, 0], v[:, 2], v[:, 1]], axis=1)

    print(f' Rotation matrix: \n{v}')
    print(f'Det of rotation matrix: {np.linalg.det(v):.3f}')
    if np.linalg.det(v) < 0.:
        raise ValueError("Determinant is negative. Manage it.")
    rotated_points = (centered_points) @ v.T
    return rotated_points, v, means


def rotate_strain_like_data(rotation, data):
    """Rotate strain-like arrays which reperesents rank-2 symmetric tensors
    with the given rotation matrix following R X R^T,
    where R is a rotation matrix and X is a rank-2 tensor.

    Args:
        rotation: numpy.ndarray
            (dim, dim) shaped matrix.
        data: numpy.ndarray
            (n, dim!) shaped array representing symmetric tensors.
    Returns:
        rotated_tensors: numpy.ndarray
            (n, dim!) shaped array.
    """
    return symmat2array(
        rotation @ array2symmat(data, from_engineering=True) @ rotation.T,
        to_engineering=True)


def read_fem(data_dir, *, return_femdata=False, read_fem_all=False,
             read_femio_npy=True):
    """Read FEM data.

    Args:
        data_dir: str
            Data directory name.
        return_femdata: bool, optional [False]
            If True, also return FEMData object.
        read_fem_all: bool, optional [False]
            If True, read FEMData all. Only effective if return_femdata is
            True.
    Returns:
        node: numpy.ndarray
            Node positions.
        disp: numpy.ndarray
            Nodal displacements.
        fem_data: femio.FEMData, optional
            FEMData object. Only provided if return_femdata is True.
    """
    fem_data = FEMData.read_directory(
        'fistr', data_dir, read_npy=read_femio_npy)
    node = fem_data.nodes.data
    if 'DISPLACEMENT' in fem_data.nodal_data:
        disp = fem_data.access_attribute('DISPLACEMENT')
    else:
        raise ValueError('Displacement not in FrontISTR data')

    if return_femdata:
        return node, disp, fem_data
    else:
        return node, disp


def align_fem(data_dir):
    _, _, fem_data = read_fem(data_dir, return_femdata=True)
    pos = fem_data.nodes.data + fem_data.access_attribute('displacement')
    aligned_pos, _, _ = align_data(pos)
    new_fem_data = FEMData(
        nodes=FEMAttribute('NODE', fem_data.nodes.ids, aligned_pos),
        elements=fem_data.elements)
    new_fem_data.write('ucd', os.path.join(data_dir, 'aligned.inp'))
    return new_fem_data


def get_top_directory():
    completed_process = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        capture_output=True, text=True)
    path = Path(completed_process.stdout.rstrip('\n'))
    return path


def pad_array(array, n):
    """Pad array to the size n.

    Args:
        array: numpy.ndarray or scipy.sparse.coomatrix
            Input array of size (m, f1, f2, ...) for numpy.ndarray or (m. m)
            for scipy.sparse.coomatrix
        n: int
            Size after padding. n should be equal to or larger than m.
    Returns:
        padded_array: numpy.ndarray or scipy.sparse.coomatrix
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
            variable for variable in variables
            if isinstance(variable, np.ndarray)],
        axis=1)
    unconcatenatable_variables = [
        variable for variable in variables
        if not isinstance(variable, np.ndarray)]
    if len(unconcatenatable_variables) == 0:
        return concatenatable_variables
    else:
        return concatenatable_variables, unconcatenatable_variables

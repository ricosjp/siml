import glob
import multiprocessing as multi
import pathlib
import random
import shutil

import femio
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import siml.prepost as prepost
import siml.setting as setting
import torch


PLOT = False

SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def conversion_function(fem_data, data_directory):
    adj = fem_data.calculate_adjacency_matrix_element()
    nadj = prepost.normalize_adjacency_matrix(adj)

    facet_fem_data = fem_data.to_first_order().to_facets()
    inc_facet2cell = fem_data.calculate_relative_incidence_metrix_element(
        facet_fem_data, minimum_n_sharing=3)
    inc_cell2facet = inc_facet2cell.T

    x_grad, y_grad, z_grad = \
        fem_data.calculate_spatial_gradient_adjacency_matrices('elemental')
    x_grad_2, y_grad_2, z_grad_2 = \
        fem_data.calculate_spatial_gradient_adjacency_matrices(
            'elemental', n_hop=2)
    global_modulus = np.mean(
        fem_data.elemental_data.get_attribute_data('modulus'), keepdims=True)

    tensor_stress = fem_data.convert_array2symmetric_matrix(
        fem_data.elemental_data.get_attribute_data('ElementalSTRESS'),
        from_engineering=False)[:, :, :, None]
    tensor_strain = fem_data.convert_array2symmetric_matrix(
        fem_data.elemental_data.get_attribute_data('ElementalSTRAIN'),
        from_engineering=True)[:, :, :, None]
    tensor_gauss_strain1 = fem_data.convert_array2symmetric_matrix(
        fem_data.elemental_data.get_attribute_data('GaussSTRAIN1'),
        from_engineering=True)[:, :, :, None]
    tensor_gauss_strain2 = fem_data.convert_array2symmetric_matrix(
        fem_data.elemental_data.get_attribute_data('GaussSTRAIN2'),
        from_engineering=True)[:, :, :, None]
    tensor_gauss_strain3 = fem_data.convert_array2symmetric_matrix(
        fem_data.elemental_data.get_attribute_data('GaussSTRAIN3'),
        from_engineering=True)[:, :, :, None]
    tensor_gauss_strain4 = fem_data.convert_array2symmetric_matrix(
        fem_data.elemental_data.get_attribute_data('GaussSTRAIN4'),
        from_engineering=True)[:, :, :, None]
    return {
        'adj': adj, 'nadj': nadj, 'global_modulus': global_modulus,
        'inc_cell2facet': inc_cell2facet, 'inc_facet2cell': inc_facet2cell,
        'x_grad': x_grad, 'y_grad': y_grad, 'z_grad': z_grad,
        'x_grad_2': x_grad_2, 'y_grad_2': y_grad_2, 'z_grad_2': z_grad_2,
        'tensor_stress': tensor_stress, 'tensor_strain': tensor_strain,
        'tensor_gauss_strain1': tensor_gauss_strain1,
        'tensor_gauss_strain2': tensor_gauss_strain2,
        'tensor_gauss_strain3': tensor_gauss_strain3,
        'tensor_gauss_strain4': tensor_gauss_strain4,
    }


def conversion_function_grad(fem_data, raw_directory=None):
    fem_data.nodes.data = fem_data.nodes.data
    node = fem_data.nodes.data

    phi = fem_data.nodal_data.get_attribute_data('phi')
    grad = fem_data.nodal_data.get_attribute_data('grad')[..., None]

    nodal_adj = fem_data.calculate_adjacency_matrix_node()
    nodal_nadj = prepost.normalize_adjacency_matrix(nodal_adj)

    nodal_surface_normal = fem_data.calculate_surface_normals(
        mode='effective')
    nodal_grad_x, nodal_grad_y, nodal_grad_z = \
        fem_data.calculate_spatial_gradient_adjacency_matrices(
            'nodal', n_hop=1, moment_matrix=True, normals=nodal_surface_normal,
            normal_weight=10., consider_volume=False)
    inversed_moment_tensor = fem_data.nodal_data.pop(
        'inversed_moment_tensors').data[..., None]
    weighted_normal = fem_data.nodal_data.pop(
        'weighted_surface_normals').data

    neumann = np.einsum('ij,ij->i', nodal_surface_normal, grad[..., 0])
    directed_neumann = np.einsum(
        'ij,i->ij', weighted_normal, neumann)[..., None]

    inc_grad, inc_int = fem_data.calculate_spatial_gradient_incidence_matrix(
        'nodal', moment_matrix=True, normals=nodal_surface_normal,
        normal_weight=10.)
    inc_inversed_moment_tensor = fem_data.nodal_data.pop(
        'inversed_moment_tensors').data[..., None]
    inc_weighted_normal = fem_data.nodal_data.pop(
        'weighted_surface_normals').data

    np.testing.assert_almost_equal(
        inversed_moment_tensor, inc_inversed_moment_tensor)
    np.testing.assert_almost_equal(
        weighted_normal, inc_weighted_normal)

    x_component = fem_data.nodes.data[:, [0]]
    grad_x_component = np.zeros(fem_data.nodes.data.shape)[..., None]
    grad_x_component[:, 0, 0] = 1.
    x_component_neumann = np.einsum(
        'ij,ij->i', nodal_surface_normal, grad_x_component[..., 0])[..., None]
    directed_x_component_neumann = np.einsum(
        'ij,i->ij',
        weighted_normal, x_component_neumann[..., 0])[..., None]

    dict_data = {
        'node': node,
        'phi': phi,
        'grad': grad,
        'directed_neumann': directed_neumann,
        'neumann': neumann[..., None],
        'x_component': x_component,
        'grad_x_component': grad_x_component,
        'x_component_neumann': x_component_neumann,
        'directed_x_component_neumann': directed_x_component_neumann,
        'nodal_nadj': nodal_nadj,
        'nodal_grad_x': nodal_grad_x,
        'nodal_grad_y': nodal_grad_y,
        'nodal_grad_z': nodal_grad_z,
        'inc_grad_x': inc_grad[0],
        'inc_grad_y': inc_grad[1],
        'inc_grad_z': inc_grad[2],
        'inc_int': inc_int,
        'inversed_moment_tensor': inversed_moment_tensor,
        'nodal_surface_normal': nodal_surface_normal[..., None],
        'nodal_weighted_normal': weighted_normal[..., None],
    }
    return dict_data


def conversion_function_heat_time_series(fem_data, raw_directory=None):
    nodal_grad_x, nodal_grad_y, nodal_grad_z = \
        fem_data.calculate_spatial_gradient_adjacency_matrices(
            'nodal', n_hop=2)
    nodal_laplacian = (
        nodal_grad_x.dot(nodal_grad_x)
        + nodal_grad_y.dot(nodal_grad_y)
        + nodal_grad_z.dot(nodal_grad_z)).tocoo() / 6

    facet_fem_data = fem_data.to_facets()
    inc_facet2cell = fem_data.calculate_relative_incidence_metrix_element(
        facet_fem_data, minimum_n_sharing=3)
    inc_cell2facet = inc_facet2cell.T
    facet_area = facet_fem_data.calculate_element_areas()
    volume = fem_data.calculate_element_volumes()
    inc_node2facet = facet_fem_data.calculate_incidence_matrix() \
        .T.astype(int)
    inc_node2cell = fem_data.calculate_incidence_matrix() \
        .T.astype(int)

    temperature = fem_data.nodal_data.get_attribute_data('TEMPERATURE')
    raw_conductivity = fem_data.elemental_data.get_attribute_data(
        'thermal_conductivity')
    elemental_conductivity = np.array([a[0][:, 0] for a in raw_conductivity])
    nodal_conductivity = fem_data.convert_elemental2nodal(
        elemental_conductivity, mode='mean')
    global_conductivity = np.mean(
        elemental_conductivity, axis=0, keepdims=True)

    dict_data = {
        f"t_{i}": t for i, t in enumerate(temperature)}
    dict_data.update({
        f"elemental_{k}":
        fem_data.convert_nodal2elemental(v, calc_average=True)
        for k, v in dict_data.items()})
    dict_data.update({
        'nodal_grad_x': nodal_grad_x,
        'nodal_grad_y': nodal_grad_y,
        'nodal_grad_z': nodal_grad_z,
        'nodal_laplacian': nodal_laplacian,
        'inc_facet2cell': inc_facet2cell,
        'inc_cell2facet': inc_cell2facet,
        'inc_node2facet': inc_node2facet,
        'inc_node2cell': inc_node2cell,
        'facet_area': facet_area,
        'volume': volume,
        'elemental_conductivity': elemental_conductivity,
        'nodal_conductivity': nodal_conductivity,
        'global_conductivity': global_conductivity})
    return dict_data


def conversion_function_heat_boundary(fem_data, raw_directory=None):

    node = fem_data.nodal_data.get_attribute_data('node')
    elemental_volume = fem_data.calculate_element_volumes(
        raise_negative_volume=False, return_abs_volume=False)

    nodal_mean_volume = fem_data.convert_elemental2nodal(
        elemental_volume, mode='mean', raise_negative_volume=False)
    nodal_effective_volume = fem_data.convert_elemental2nodal(
        elemental_volume, mode='effective', raise_negative_volume=False)

    nodal_surface_normal = fem_data.calculate_surface_normals(
        mode='effective')
    nodal_grad_x_1, nodal_grad_y_1, nodal_grad_z_1 = \
        fem_data.calculate_spatial_gradient_adjacency_matrices(
            'nodal', n_hop=1, moment_matrix=True,
            normals=nodal_surface_normal,
            consider_volume=False, normal_weight=10.)
    inversed_moment_tensors_1 = fem_data.nodal_data.pop(
        'inversed_moment_tensors').data[..., None]
    weighted_surface_normal_1 = fem_data.nodal_data.pop(
        'weighted_surface_normals').data[..., None]

    inc_grad, inc_int = fem_data.calculate_spatial_gradient_incidence_matrix(
        'nodal', moment_matrix=True, normals=nodal_surface_normal,
        normal_weight=10.)
    inc_inversed_moment_tensor = fem_data.nodal_data.pop(
        'inversed_moment_tensors').data[..., None]
    inc_weighted_normal = fem_data.nodal_data.pop(
        'weighted_surface_normals').data[..., None]

    np.testing.assert_almost_equal(
        inversed_moment_tensors_1, inc_inversed_moment_tensor)
    np.testing.assert_almost_equal(
        weighted_surface_normal_1, inc_weighted_normal)

    dict_data = {
        'node': node,
        'nodal_mean_volume': nodal_mean_volume,
        'elemental_volume': elemental_volume,
        'nodal_effective_volume': nodal_effective_volume,
        'nodal_grad_x_1': nodal_grad_x_1,
        'nodal_grad_y_1': nodal_grad_y_1,
        'nodal_grad_z_1': nodal_grad_z_1,
        'inc_grad_x': inc_grad[0],
        'inc_grad_y': inc_grad[1],
        'inc_grad_z': inc_grad[2],
        'inc_int': inc_int,
        'inversed_moment_tensors_1': inversed_moment_tensors_1,
        'weighted_surface_normal_1': weighted_surface_normal_1,
    }

    raw_conductivity = fem_data.elemental_data.get_attribute_data(
        'thermal_conductivity_full')
    elemental_thermal_conductivity_array = np.stack([
        c[0, :-1] for c in raw_conductivity[:, 0]])
    elemental_thermal_conductivity \
        = fem_data.convert_array2symmetric_matrix(
            elemental_thermal_conductivity_array,
            from_engineering=False)[:, :, :, None]
    nodal_thermal_conductivity_array \
        = fem_data.convert_elemental2nodal(
            elemental_thermal_conductivity_array, mode='mean',
            raise_negative_volume=False)
    nodal_thermal_conductivity \
        = fem_data.convert_array2symmetric_matrix(
            nodal_thermal_conductivity_array, from_engineering=False)[
                :, :, :, None]

    nodal_t_0 = fem_data.nodal_data.get_attribute_data(
        'INITIAL_TEMPERATURE')
    global_thermal_conductivity = np.mean(
        elemental_thermal_conductivity, keepdims=True, axis=0)

    dict_data.update({
        'nodal_thermal_conductivity': nodal_thermal_conductivity,
        'nodal_t_0': nodal_t_0,
        'global_thermal_conductivity': global_thermal_conductivity,
    })
    temperatures = fem_data.nodal_data.get_attribute_data(
        'TEMPERATURE')
    if len(temperatures.shape) != 3:
        raise ValueError(
            'Temperature is not time series '
            f"(shape: {temperatures.shape}).\n"
            'Set conversion.time_series = true in the YAML file.')
    if 'time_steps' not in fem_data.settings:
        raise ValueError(fem_data.settings)
    dict_t_data = {
        f"nodal_t_{step}": t for step, t in zip(
            fem_data.settings['time_steps'], temperatures)}
    max_timestep = max(fem_data.settings['time_steps'])
    dict_data.update(dict_t_data)
    dict_data.update({
        'nodal_t_diff':
        dict_data[f"nodal_t_{max_timestep}"] - dict_data['nodal_t_0']})

    nodal_adj = fem_data.calculate_adjacency_matrix_node()
    nodal_nadj = prepost.normalize_adjacency_matrix(nodal_adj)
    dict_data.update({
        'nodal_adj': nodal_adj, 'nodal_nadj': nodal_nadj,
        'nodal_grad_x_1': nodal_grad_x_1,
        'nodal_grad_y_1': nodal_grad_y_1,
        'nodal_grad_z_1': nodal_grad_z_1,
        'ts_temperature': temperatures,
        'ts_mean_temperature': np.mean(temperatures, axis=1, keepdims=True),
        'inversed_moment_tensors_1': inversed_moment_tensors_1,
        'weighted_surface_normal_1': weighted_surface_normal_1,
        'nodal_surface_normal': nodal_surface_normal[..., None],
    })

    dict_data.update(_extract_boundary(fem_data, dict_data))
    return dict_data


def conversion_function_heat_interaction(fem_data, raw_directory=None):

    dict_data = {}
    dict_data.update(_load_mesh(raw_directory / 'mesh_1.inp', '_1'))
    dict_data.update(_load_mesh(raw_directory / 'mesh_2.inp', '_2'))
    dict_data.update({
        'incidence_2to1':
        sp.load_npz('tests/data/heat_interaction/raw/incidence_2to1.npz'),
        'periodic_2':
        sp.load_npz('tests/data/heat_interaction/raw/periodic_2.npz'),
        'coeff':
        np.load(raw_directory / 'coeff.npy')[None, None],
        'heat_transfer':
        np.load(raw_directory / 'heat_transfer.npy')[None, None],
    })

    return dict_data


def _load_mesh(path, suffix):
    fem_data = femio.read_files('ucd', path)

    nodal_surface_normal = fem_data.calculate_surface_normals(
        mode='effective')
    nodal_grad_x_1, nodal_grad_y_1, nodal_grad_z_1 = \
        fem_data.calculate_spatial_gradient_adjacency_matrices(
            'nodal', n_hop=1, moment_matrix=True,
            normals=nodal_surface_normal,
            consider_volume=False, normal_weight_factor=1.)
    inversed_moment_tensors_1 = fem_data.nodal_data.get_attribute_data(
        'inversed_moment_tensors')[..., None]
    weighted_surface_normal_1 = fem_data.nodal_data.get_attribute_data(
        'weighted_surface_normals')[..., None]
    adj = fem_data.calculate_adjacency_matrix_node()
    nadj = prepost.normalize_adjacency_matrix(adj)

    phi_0 = fem_data.nodal_data.get_attribute_data('phi_0')
    phi_1 = fem_data.nodal_data.get_attribute_data('phi_1')

    return {
        f"adj{suffix}": adj,
        f"nadj{suffix}": nadj,
        f"gx{suffix}": nodal_grad_x_1,
        f"gy{suffix}": nodal_grad_y_1,
        f"gz{suffix}": nodal_grad_z_1,
        f"minv{suffix}": inversed_moment_tensors_1,
        f"wnorm{suffix}": weighted_surface_normal_1,
        f"phi_0{suffix}": phi_0,
        f"phi_1{suffix}": phi_1,
    }


def _extract_boundary(fem_data, dict_data):
    dirichlet = np.ones((len(fem_data.nodes), 1)) * np.nan
    padded_dirichlet = np.zeros((len(fem_data.nodes), 1))
    dirichlet_label = np.zeros((len(fem_data.nodes), 1))
    neumann = np.zeros((len(fem_data.nodes), 1))
    directed_neumann = np.zeros((len(fem_data.nodes), 3, 1))

    if 'fixtemp' in fem_data.constraints:
        fixtemp = fem_data.constraints['fixtemp']
        if len(fixtemp) > 0:
            dirichlet_indices = fem_data.nodes.ids2indices(fixtemp.ids)
            dirichlet[dirichlet_indices, 0] = fixtemp.data
            padded_dirichlet[dirichlet_indices, 0] = fixtemp.data
            dirichlet_label[dirichlet_indices, 0] = 1

    if 'pure_cflux' in fem_data.constraints:
        pure_cflux = fem_data.constraints['pure_cflux']
        surface_fem_data = fem_data.to_surface()
        nodal_areas = surface_fem_data.convert_elemental2nodal(
            surface_fem_data.calculate_element_areas(), mode='effective')
        if len(pure_cflux) > 0:
            pure_cflux_indices = fem_data.nodes.ids2indices(pure_cflux.ids)
            neumann[pure_cflux_indices, 0] = - pure_cflux.data[:, 0] \
                / nodal_areas[surface_fem_data.nodes.ids2indices(
                    pure_cflux.ids), 0]
            directed_neumann = np.einsum(
                'ij,i->ij', dict_data['nodal_surface_normal'][..., 0],
                neumann[:, 0])[..., None]

    return {
        'dirichlet': dirichlet, 'dirichlet_label': dirichlet_label,
        'padded_dirichlet': padded_dirichlet,
        'neumann': neumann, 'directed_neumann': directed_neumann}


def conversion_function_rotation_thermal_stress(fem_data, raw_directory=None):
    adj = fem_data.calculate_adjacency_matrix_node()
    nadj = prepost.normalize_adjacency_matrix(adj)
    nodal_grad_x, nodal_grad_y, nodal_grad_z = \
        fem_data.calculate_spatial_gradient_adjacency_matrices(
            'nodal', n_hop=2)
    nodal_hess_xx = nodal_grad_x.dot(nodal_grad_x).tocoo()
    nodal_hess_xy = nodal_grad_x.dot(nodal_grad_y).tocoo()
    nodal_hess_xz = nodal_grad_x.dot(nodal_grad_z).tocoo()
    nodal_hess_yx = nodal_grad_y.dot(nodal_grad_x).tocoo()
    nodal_hess_yy = nodal_grad_y.dot(nodal_grad_y).tocoo()
    nodal_hess_yz = nodal_grad_y.dot(nodal_grad_z).tocoo()
    nodal_hess_zx = nodal_grad_z.dot(nodal_grad_x).tocoo()
    nodal_hess_zy = nodal_grad_z.dot(nodal_grad_y).tocoo()
    nodal_hess_zz = nodal_grad_z.dot(nodal_grad_z).tocoo()

    frame_adjs = fem_data.calculate_frame_tensor_adjs(mode='nodal', n_hop=2)
    nodal_frame_xx = frame_adjs[0][0]
    nodal_frame_xy = frame_adjs[0][1]
    nodal_frame_xz = frame_adjs[0][2]
    nodal_frame_yx = frame_adjs[1][0]
    nodal_frame_yy = frame_adjs[1][1]
    nodal_frame_yz = frame_adjs[1][2]
    nodal_frame_zx = frame_adjs[2][0]
    nodal_frame_zy = frame_adjs[2][1]
    nodal_frame_zz = frame_adjs[2][2]

    filter_ = fem_data.filter_first_order_nodes()

    node = fem_data.nodes.data[filter_]
    nodal_mean_volume = fem_data.convert_elemental2nodal(
        fem_data.calculate_element_volumes(), mode='mean')
    nodal_concentrated_volume = fem_data.convert_elemental2nodal(
        fem_data.calculate_element_volumes(), mode='effective')
    initial_temperature = fem_data.nodal_data.get_attribute_data(
        'INITIAL_TEMPERATURE')[filter_]
    cnt_temperature = fem_data.nodal_data.get_attribute_data(
        'CNT_TEMPERATURE')[filter_]

    elemental_lte_array = fem_data.elemental_data.get_attribute_data(
        'linear_thermal_expansion_coefficient_full')
    nodal_lte_array = fem_data.convert_elemental2nodal(
        elemental_lte_array, mode='mean')
    global_lte_array = np.mean(
        elemental_lte_array, axis=0, keepdims=True)

    elemental_lte_mat = fem_data.convert_array2symmetric_matrix(
        elemental_lte_array, from_engineering=True)
    nodal_lte_mat = fem_data.convert_array2symmetric_matrix(
        nodal_lte_array, from_engineering=True)
    global_lte_mat = np.mean(
        elemental_lte_mat, axis=0, keepdims=True)

    elemental_strain_array = fem_data.elemental_data.get_attribute_data(
        'ElementalSTRAIN')
    nodal_strain_array = fem_data.nodal_data.get_attribute_data(
        'NodalSTRAIN')[filter_]
    elemental_strain_mat = fem_data.convert_array2symmetric_matrix(
        elemental_strain_array, from_engineering=True)
    nodal_strain_mat = fem_data.convert_array2symmetric_matrix(
        nodal_strain_array, from_engineering=True)

    inc_grad, inc_int = fem_data.calculate_spatial_gradient_incidence_matrix(
        'nodal', moment_matrix=True, normals=False)
    inversed_moment_tensors = fem_data.nodal_data.pop(
        'inversed_moment_tensors').data[..., None]

    dict_data = {
        'nadj': nadj,
        'nodal_grad_x': nodal_grad_x,
        'nodal_grad_y': nodal_grad_y,
        'nodal_grad_z': nodal_grad_z,
        'inc_grad_x': inc_grad[0],
        'inc_grad_y': inc_grad[1],
        'inc_grad_z': inc_grad[2],
        'inc_int': inc_int,
        'minv': inversed_moment_tensors,
        'nodal_hess_xx': nodal_hess_xx,
        'nodal_hess_xy': nodal_hess_xy,
        'nodal_hess_xz': nodal_hess_xz,
        'nodal_hess_yx': nodal_hess_yx,
        'nodal_hess_yy': nodal_hess_yy,
        'nodal_hess_yz': nodal_hess_yz,
        'nodal_hess_zx': nodal_hess_zx,
        'nodal_hess_zy': nodal_hess_zy,
        'nodal_hess_zz': nodal_hess_zz,
        'nodal_frame_xx': nodal_frame_xx,
        'nodal_frame_xy': nodal_frame_xy,
        'nodal_frame_xz': nodal_frame_xz,
        'nodal_frame_yx': nodal_frame_yx,
        'nodal_frame_yy': nodal_frame_yy,
        'nodal_frame_yz': nodal_frame_yz,
        'nodal_frame_zx': nodal_frame_zx,
        'nodal_frame_zy': nodal_frame_zy,
        'nodal_frame_zz': nodal_frame_zz,
        'node': node,
        'nodal_strain_array': nodal_strain_array,
        'elemental_strain_array': elemental_strain_array,
        'nodal_strain_mat': nodal_strain_mat[..., None],
        'elemental_strain_mat': elemental_strain_mat[..., None],
        'nodal_mean_volume': nodal_mean_volume,
        'nodal_concentrated_volume': nodal_concentrated_volume,
        'initial_temperature': initial_temperature,
        'cnt_temperature': cnt_temperature,
        'elemental_lte_array': elemental_lte_array,
        'nodal_lte_array': nodal_lte_array,
        'global_lte_array': global_lte_array,
        'elemental_lte_mat': elemental_lte_mat[..., None],
        'nodal_lte_mat': nodal_lte_mat[..., None],
        'global_lte_mat': global_lte_mat[..., None],
    }
    return dict_data


def preprocess_deform():
    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/deform/data.yml'))

    raw_converter = prepost.RawConverter(
        main_setting, recursive=True, force_renew=True,
        conversion_function=conversion_function)
    raw_converter.convert()

    preprocessor = prepost.Preprocessor(main_setting, force_renew=True)
    preprocessor.preprocess_interim_data()

    interim_path = pathlib.Path(
        'tests/data/deform/interim/train/tet2_3_modulusx0.9000')
    preprocessed_path = pathlib.Path(
        'tests/data/deform/preprocessed/train/tet2_3_modulusx0.9000')
    interim_x_grad = sp.load_npz(interim_path / 'x_grad.npz')
    preprocessed_x_grad = sp.load_npz(preprocessed_path / 'x_grad.npz')
    scale_x_grad = preprocessed_x_grad.data / interim_x_grad.data

    dict_reference_scale = calculate_scale_isoam()
    np.testing.assert_almost_equal(np.var(scale_x_grad), 0.)
    np.testing.assert_almost_equal(
        np.mean(scale_x_grad), dict_reference_scale['x_grad'])

    interim_y_grad = sp.load_npz(interim_path / 'y_grad.npz')
    preprocessed_y_grad = sp.load_npz(preprocessed_path / 'y_grad.npz')
    scale_y_grad = preprocessed_y_grad.data / interim_y_grad.data
    np.testing.assert_almost_equal(
        np.mean(scale_y_grad), np.mean(scale_x_grad))

    return


def preprocess_grad():
    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/grad/data.yml'))

    raw_converter = prepost.RawConverter(
        main_setting, recursive=True, force_renew=True,
        conversion_function=conversion_function_grad)
    raw_converter.convert()

    preprocessor = prepost.Preprocessor(main_setting, force_renew=True)
    preprocessor.preprocess_interim_data()

    return


def calculate_scale_isoam():
    data_directory = pathlib.Path('tests/data/deform/interim')

    if not data_directory.is_dir():
        raise ValueError(f"Directory does not exist: {data_directory}")

    data_directories = [
        pathlib.Path(f).parent for f in glob.glob(
            str(data_directory / '**/nadj.npz'),
            recursive=True)]
    max_process = 2
    chunksize = max(
        len(data_directories) // max_process // 16, 1)

    with multi.Pool(max_process) as pool:
        grad_stats = pool.map(
            calculate_grad_stats, data_directories, chunksize=chunksize)
    dict_scales = summarize_scales(grad_stats)
    return dict_scales


def summarize_scales(grad_stats):
    grad_keys = grad_stats[0].keys()
    dict_n = {
        k: np.sum(np.fromiter(
            (grad_stat[k][0] for grad_stat in grad_stats), int))
        for k in grad_keys}
    dict_sum = {
        k: np.sum(np.fromiter(
            (grad_stat[k][1] for grad_stat in grad_stats), float))
        for k in grad_keys}
    dict_scales = {
        k: 1 / float(dict_sum[k] / dict_n[k])**.5 for k in grad_keys}
    return dict_scales


def calculate_grad_stats(input_directory):
    print(f"Processing: {input_directory}")
    grad_x_files = glob.glob(str(input_directory / 'x_grad.npz'))
    return {
        pathlib.Path(grad_x_file).stem.replace('_x', ''):
        _calculate_stat(pathlib.Path(grad_x_file))
        for grad_x_file in grad_x_files}


def _calculate_stat(grad_x_file):
    grad_x = sp.load_npz(grad_x_file)
    parent = grad_x_file.parent
    grad_y = sp.load_npz(parent / grad_x_file.name.replace('x', 'y'))
    grad_z = sp.load_npz(parent / grad_x_file.name.replace('x', 'z'))
    n = grad_x.shape[0]
    scale = np.sum(
        grad_x.diagonal()**2 + grad_y.diagonal()**2 + grad_z.diagonal()**2)
    return n, scale


def preprocess_rotation_thermal_stress():
    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/rotation_thermal_stress/data.yml'))

    raw_converter = prepost.RawConverter(
        main_setting, recursive=True, force_renew=True,
        to_first_order=True,
        conversion_function=conversion_function_rotation_thermal_stress)
    raw_converter.convert()

    preprocessor = prepost.Preprocessor(main_setting, force_renew=True)
    preprocessor.preprocess_interim_data()
    return


def preprocess_heat_time_series():
    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/heat_time_series/data.yml'))

    raw_converter = prepost.RawConverter(
        main_setting, recursive=True, force_renew=True,
        conversion_function=conversion_function_heat_time_series,
        write_ucd=False)
    raw_converter.convert()

    preprocessor = prepost.Preprocessor(main_setting, force_renew=True)
    preprocessor.preprocess_interim_data()
    return


def preprocess_heat_boundary():
    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/heat_boundary/data.yml'))

    raw_converter = prepost.RawConverter(
        main_setting, recursive=True, force_renew=True,
        conversion_function=conversion_function_heat_boundary,
        write_ucd=False)
    raw_converter.convert()

    preprocessor = prepost.Preprocessor(main_setting, force_renew=True)
    preprocessor.preprocess_interim_data()
    return


def preprocess_heat_interaction():
    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/heat_interaction/data.yml'))

    raw_converter = prepost.RawConverter(
        main_setting, recursive=True, force_renew=True,
        conversion_function=conversion_function_heat_interaction,
        write_ucd=False)
    raw_converter.convert()

    preprocessor = prepost.Preprocessor(main_setting, force_renew=True)
    preprocessor.preprocess_interim_data()
    return


def preprocess_deform_timeseries():
    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/deform_timeseries/data.yml'))

    preprocessor = prepost.Preprocessor(main_setting, force_renew=True)
    preprocessor.preprocess_interim_data()
    return


def rotation_conversion_function(fem_data, raw_directory):
    nodal_mean_volume = fem_data.convert_elemental2nodal(
        fem_data.calculate_element_volumes(), mode='mean')
    nodal_concentrated_volume = fem_data.convert_elemental2nodal(
        fem_data.calculate_element_volumes(), mode='effective')

    nodal_grad_x, nodal_grad_y, nodal_grad_z = \
        fem_data.calculate_spatial_gradient_adjacency_matrices(
            'nodal', n_hop=2)
    nodal_laplacian = (
        nodal_grad_x.dot(nodal_grad_x)
        + nodal_grad_y.dot(nodal_grad_y)
        + nodal_grad_z.dot(nodal_grad_z)).tocoo() / 6
    node = fem_data.nodes.data
    t_init = fem_data.nodal_data.get_attribute_data('t_init')
    ucd_data = femio.FEMData.read_files(
        'ucd', [raw_directory / 'mesh_vis_psf.0100.inp'])
    t_100 = ucd_data.nodal_data.get_attribute_data('TEMPERATURE')
    return {
        'nodal_mean_volume': nodal_mean_volume,
        'nodal_concentrated_volume': nodal_concentrated_volume,
        'nodal_grad_x': nodal_grad_x,
        'nodal_grad_y': nodal_grad_y,
        'nodal_grad_z': nodal_grad_z,
        'nodal_laplacian': nodal_laplacian,
        'node': node, 't_init': t_init, 't_100': t_100}


def preprocess_rotation():
    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/rotation/data.yml'))

    raw_converter = prepost.RawConverter(
        main_setting, recursive=True, force_renew=True,
        conversion_function=rotation_conversion_function)
    raw_converter.convert()

    preprocessor = prepost.Preprocessor(main_setting, force_renew=True)
    preprocessor.preprocess_interim_data()
    return


def generate_linear():
    n_element = 200

    def generate_data(root_dir, n_data):
        if root_dir.exists():
            shutil.rmtree(root_dir)
        for i in range(n_data):
            x1 = np.random.rand(n_element, 2)
            x2 = np.random.rand(n_element, 1)
            y = 2 * x1 + 3 * x2 + 10.

            output_directory = root_dir / f"{i}"
            output_directory.mkdir(parents=True)
            np.save(output_directory / 'x1.npy', x1.astype(np.float32))
            np.save(output_directory / 'x2.npy', x2.astype(np.float32))
            np.save(output_directory / 'y.npy', y.astype(np.float32))
            (output_directory / 'converted').touch()
        return

    output_root = pathlib.Path('tests/data/linear/interim')
    train_root = output_root / 'train'
    n_train_data = 5
    generate_data(train_root, n_train_data)

    validation_root = output_root / 'validation'
    n_validation_data = 2
    generate_data(validation_root, n_validation_data)

    test_root = output_root / 'test'
    n_test_data = 2
    generate_data(test_root, n_test_data)

    p = prepost.Preprocessor.read_settings(
        pathlib.Path('tests/data/linear/linear.yml'), force_renew=True)
    p.preprocess_interim_data()

    return


def generate_ode():
    time_range = (10., 50.)
    delta_t = .1

    def f0(ts, xs):
        ys = np.zeros(list(xs.shape[:2]) + [1])
        ys[0] = np.random.rand(*list(ys.shape)[1:])
        for i in range(1, len(ts)):
            ys[i, :, 0] = ys[i - 1, :, 0] + delta_t * (- .1 * ys[i - 1, :, 0])
        return ys

    def f1(ts, xs):
        ys = np.zeros(list(xs.shape[:2]) + [1])
        ys[0] = np.random.rand(*list(ys.shape)[1:])
        for i in range(1, len(ts)):
            ys[i, :, 0] = ys[i - 1, :, 0] + delta_t * xs[i, :, 0] * .1
        return ys

    def f2(ts, xs):
        ys = np.zeros(list(xs.shape[:2]) + [1])
        ys[0] = np.random.rand(*list(ys.shape)[1:])
        for i in range(1, len(ts)):
            ys[i, :, 0] = ys[i - 1, :, 0] + delta_t * (
                .01 * xs[i, :, 1] - .01 * xs[i, :, 0] * xs[i, :, 3]
                - .01 * ys[i - 1, :, 0])
        return ys

    def f3(ts, xs):
        ys = np.zeros(list(xs.shape[:2]) + [2])
        ys[0] = np.random.rand(*list(ys.shape)[1:]) * 2 - 1
        for i in range(1, len(ts)):
            ys[i, :, 0] = ys[i - 1, :, 0] + delta_t * (
                - .05 * ys[i - 1, :, 1]
                + .01 * (1 - ys[i - 1, :, 1]**2) * ys[i - 1, :, 0])
            ys[i, :, 1] = ys[i - 1, :, 1] + delta_t * ys[i - 1, :, 0]
        return ys

    def generate_ode(root_dir, n_data):
        if root_dir.exists():
            shutil.rmtree(root_dir)

        for i in range(n_data):
            n_element = np.random.randint(3, 10)
            t_max = np.random.rand() * (
                time_range[1] - time_range[0]) + time_range[0]
            ts = np.arange(0., t_max, delta_t)
            x0 = np.random.rand() * np.sin(
                2 * np.pi * (np.random.rand() / 10. * ts + np.random.rand()))
            x1 = np.random.rand() * np.sin(
                2 * np.pi * (np.random.rand() / 20. * ts + np.random.rand()))
            x2 = np.random.rand() * (
                1 - np.exp(- ts / 5. * np.random.rand())) + np.random.rand()
            x3 = np.exp(- ts / 10. * np.random.rand()) + np.random.rand()
            _xs = np.stack([x0, x1, x2, x3], axis=1)[:, None, :]
            xs = np.concatenate([
                _xs * a for a in np.linspace(1., 2., n_element)], axis=1)

            y0 = f0(ts, xs)
            y1 = f1(ts, xs)
            y2 = f2(ts, xs)
            y3 = f3(ts, xs)

            stacked_ts = np.concatenate(
                [ts[:, None, None]] * n_element, axis=1)

            output_directory = root_dir / f"{i}"
            output_directory.mkdir(parents=True)
            np.save(output_directory / 't.npy', stacked_ts.astype(np.float32))
            np.save(output_directory / 'x.npy', xs.astype(np.float32))
            np.save(output_directory / 'y0.npy', y0.astype(np.float32))
            np.save(output_directory / 'y1.npy', y1.astype(np.float32))
            np.save(output_directory / 'y2.npy', y2.astype(np.float32))
            np.save(output_directory / 'y3.npy', y3.astype(np.float32))
            np.save(
                output_directory / 'y0_initial.npy',
                (np.ones(y0.shape) * y0[0, :, :]).astype(np.float32))
            np.save(
                output_directory / 'y1_initial.npy',
                (np.ones(y1.shape) * y1[0, :, :]).astype(np.float32))
            np.save(
                output_directory / 'y2_initial.npy',
                (np.ones(y2.shape) * y2[0, :, :]).astype(np.float32))
            np.save(
                output_directory / 'y3_initial.npy',
                (np.ones(y3.shape) * y3[0, :, :]).astype(np.float32))
            (output_directory / 'converted').touch()

            if PLOT:
                plt.plot(ts, x0, label='x0')
                plt.plot(ts, x1, label='x1')
                plt.plot(ts, x2, label='x2')
                plt.plot(ts, x3, label='x3')
                plt.plot(ts, y0[:, 0, 0], label='y0')
                plt.plot(ts, y1[:, 0, 0], label='y1')
                plt.plot(ts, y2[:, 0, 0], label='y2')
                plt.plot(ts, y3[:, 0, 0], label='y3-0')
                plt.plot(ts, y3[:, 0, 1], label='y3-1')
                plt.legend()
                plt.savefig(output_directory / 'plot.pdf')
                plt.show()
        return

    generate_ode(pathlib.Path('tests/data/ode/interim/train'), 100)
    generate_ode(pathlib.Path('tests/data/ode/interim/validation'), 10)
    generate_ode(pathlib.Path('tests/data/ode/interim/test'), 10)

    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/ode/data.yml'))
    preprocessor = prepost.Preprocessor(main_setting, force_renew=True)
    preprocessor.preprocess_interim_data()
    return


def generate_large():
    n_feat = 10
    n_element = 2000

    def generate_data(root_dir, n_data):
        if root_dir.exists():
            shutil.rmtree(root_dir)
        for i in range(n_data):
            r1 = np.random.rand()
            r2 = np.random.rand()
            floor = min(r1, r2)
            ceil = max(r1, r2)

            x = np.random.rand(n_element, n_feat) * (ceil - floor) + floor
            y = np.sin(x * 4. * np.pi)

            output_directory = root_dir / f"{i}"
            output_directory.mkdir(parents=True)
            np.save(output_directory / 'x.npy', x.astype(np.float32))
            np.save(output_directory / 'y.npy', y.astype(np.float32))

    output_root = pathlib.Path('tests/data/large/preprocessed')
    train_root = output_root / 'train'
    n_train_data = 20
    generate_data(train_root, n_train_data)

    validation_root = output_root / 'validation'
    n_validation_data = 2
    generate_data(validation_root, n_validation_data)

    test_root = output_root / 'test'
    n_test_data = 2
    generate_data(test_root, n_test_data)
    return


if __name__ == '__main__':
    preprocess_heat_boundary()
    preprocess_heat_interaction()
    preprocess_grad()
    preprocess_deform()
    preprocess_rotation_thermal_stress()
    preprocess_heat_time_series()
    generate_ode()
    preprocess_deform_timeseries()
    preprocess_rotation()
    # generate_linear()
    generate_large()

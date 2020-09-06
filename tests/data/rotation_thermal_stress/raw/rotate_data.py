import argparse
import glob
import pathlib
import shutil
import subprocess

import femio
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root_data_directory',
        type=pathlib.Path,
        help='Root of the data directory which contains thermal analysis')
    parser.add_argument(
        '-n', '--n-trial',
        type=int,
        default=2,
        help='The number of trial [2]')
    parser.add_argument(
        '-f', '--file-basename',
        type=str,
        default='thermal',
        help='Base name of output files [''thermal'']')
    args = parser.parse_args()

    original_data_paths = [
        pathlib.Path(cnt_file).parent for cnt_file
        in glob.glob(
            str(args.root_data_directory / '**/thermal.cnt'),
            recursive=True)]

    for original_data_path in original_data_paths:
        transform_data(
            original_data_path, additional_trial=args.n_trial,
            basename=args.file_basename)
    return


def transform_data(data_path, *, additional_trial=2, basename='thermal'):
    fem_data = femio.FEMData.read_directory('fistr', data_path, read_npy=False)

    # Simple rotation
    orthogonal_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    translation_vector = np.array([0., 0., 0.])
    output_directory = data_path.parent / (
        data_path.name + '_transformed_rotation_yz')
    process(fem_data, orthogonal_matrix, translation_vector, output_directory)

    # Simple mirror
    orthogonal_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    translation_vector = np.array([0., 0., 0.])
    output_directory = data_path.parent / (
        data_path.name + '_transformed_mirror_xy')
    process(
        fem_data, orthogonal_matrix, translation_vector, output_directory,
        basename=basename)

    for i in range(additional_trial):
        orthogonal_matrix = generate_orthogonal_matrix()
        translation_vector = (2 * np.random.rand(3) - 1.) * 10.
        output_directory = data_path.parent / (
            data_path.name + f"_transformed_{i}")
        process(
            fem_data, orthogonal_matrix, translation_vector, output_directory,
            basename=basename)
    return


def process(
        fem_data, orthogonal_matrix, translation_vector, output_directory,
        basename='thermal'):
    _transform_data(
        fem_data, orthogonal_matrix, translation_vector, output_directory,
        basename)
    np.savetxt(output_directory / 'orthogonal_matrix.txt', orthogonal_matrix)
    np.savetxt(
        output_directory / 'det.txt', np.linalg.det(orthogonal_matrix)[None])
    np.savetxt(output_directory / 'translation_vector.txt', translation_vector)
    sp = subprocess.run(
        f"cd {output_directory} && fistr1", shell=True, check=True)
    print(sp)
    validate_results(
        fem_data, output_directory, orthogonal_matrix, translation_vector)
    return


def generate_orthogonal_matrix():
    vec1 = normalize(np.random.rand(3)*2 - 1)
    vec2 = normalize(np.random.rand(3)*2 - 1)
    vec3 = normalize(np.cross(vec1, vec2))
    vec2 = np.cross(vec3, vec1)
    return np.array([
        vec1 * np.random.choice([-1, 1]),  # det = -1 or 1
        vec2 * np.random.choice([-1, 1]),  # det = -1 or 1
        vec3 * np.random.choice([-1, 1]),  # det = -1 or 1
    ])


def normalize(x):
    return x / np.linalg.norm(x)


def validate_results(
        original_fem_data, output_directory, orthogonal_matrix,
        translation_vector):
    calculated_fem_data = femio.FEMData.read_directory(
        'fistr', output_directory, read_npy=False)
    transformed_original_strain = transform_tensor_array(
        original_fem_data,
        original_fem_data.elemental_data.get_attribute_data(
            'ElementalSTRAIN'), orthogonal_matrix)
    calculated_strain = calculated_fem_data.elemental_data.get_attribute_data(
        'ElementalSTRAIN')
    mean_rmse = np.mean((
        transformed_original_strain - calculated_strain)**2)**.5
    ref = np.mean(transformed_original_strain**2)**.5
    relative_error_percent = mean_rmse / ref * 100
    print('========================')
    print(f"mean error: {mean_rmse}")
    print(f"relative mean error: {relative_error_percent}")
    print('========================')
    with open(output_directory / 'log.txt', 'w') as f:
        f.write(f"mean error: {mean_rmse}\n")
        f.write(f"relative mean error: {relative_error_percent}\n")
    transformed_original_strain_mat \
        = original_fem_data.convert_array2symmetric_matrix(
            transformed_original_strain[:5], from_engineering=True)
    calculated_strain_mat = calculated_fem_data.convert_array2symmetric_matrix(
        calculated_strain[:5], from_engineering=True)
    if relative_error_percent > 1e-5:
        print(transformed_original_strain_mat)
        print(calculated_strain_mat)
        print(transformed_original_strain_mat - calculated_strain_mat)
        raise ValueError(
            f"Error too big for: {output_directory}\n"
            f"Matrix: {orthogonal_matrix}\n"
            f"Det: {np.linalg.det(orthogonal_matrix)}\n"
            f"Translation: {translation_vector}\n"
        )
    return


def transform_tensor_array(fem_data, tensor_array, orthogonal_matrix):
    symmetric_mat = fem_data.convert_array2symmetric_matrix(
        tensor_array, from_engineering=True)
    transformed_mat = np.array([
        orthogonal_matrix @ m @ orthogonal_matrix.T for m in symmetric_mat])
    transformed_array = fem_data.convert_symmetric_matrix2array(
        transformed_mat, to_engineering=True)
    return transformed_array


def _transform_data(
        fem_data, orthogonal_matrix, translation_vector, output_directory,
        basename):
    shutil.rmtree(output_directory, ignore_errors=True)

    # Node
    original_node = fem_data.nodal_data.get_attribute_data('NODE')
    transformed_nodes = np.array([
        orthogonal_matrix @ n for n in original_node]) + translation_vector

    # Element
    e = fem_data.elements.data
    if np.linalg.det(orthogonal_matrix) < 0:
        new_e = np.stack([
            e[:, 1],
            e[:, 0],
            e[:, 2],
            e[:, 3],
            e[:, 5],
            e[:, 4],
            e[:, 6],
            e[:, 8],
            e[:, 7],
            e[:, 9],
        ], axis=1)
    else:
        new_e = e

    new_fem_data = femio.FEMData(
        femio.FEMAttribute(
            'NODE', ids=fem_data.nodes.ids, data=transformed_nodes),
        femio.FEMElementalAttribute(
            'ELEMENT', data={'tet2': femio.FEMAttribute(
                'tet2', fem_data.elements.ids, new_e)}))
    # Confirm volume is positive
    new_fem_data.calculate_element_volumes(raise_negative_volume=True)

    # Nodal data
    original_t_init = fem_data.nodal_data.get_attribute_data(
        'INITIAL_TEMPERATURE')
    original_t_cnt = fem_data.nodal_data.get_attribute_data('CNT_TEMPERATURE')
    nodal_data_dict = {
        'INITIAL_TEMPERATURE': original_t_init,
        'CNT_TEMPERATURE': original_t_cnt}
    new_fem_data.nodal_data.update_data(
        new_fem_data.nodes.ids, nodal_data_dict)

    # Material data
    original_poisson_ratio = np.mean(
        fem_data.elemental_data.get_attribute_data('Poisson_ratio'),
        axis=0, keepdims=True)
    original_young_modulus = np.mean(
        fem_data.elemental_data.get_attribute_data('Young_modulus'),
        axis=0, keepdims=True)
    original_ltec = np.mean(
        fem_data.elemental_data.get_attribute_data(
            'linear_thermal_expansion_coefficient_full'),
        axis=0, keepdims=True)
    transformed_ltec_array = transform_tensor_array(
        new_fem_data, original_ltec, orthogonal_matrix)
    material_data_dict = {
        'Poisson_ratio': original_poisson_ratio,
        'Young_modulus': original_young_modulus,
        'linear_thermal_expansion_coefficient_full': transformed_ltec_array}
    new_fem_data.materials.update_data('MAT_ALL', material_data_dict)

    # Elemental data
    n_element = len(new_fem_data.elements.ids)
    elemental_data_dict = {
        'Poisson_ratio': original_poisson_ratio * np.ones((n_element, 1)),
        'Young_modulus': original_young_modulus * np.ones((n_element, 1)),
        'linear_thermal_expansion_coefficient_full':
        transformed_ltec_array * np.ones((n_element, 6))}
    new_fem_data.elemental_data.update_data(
        new_fem_data.elements.ids, elemental_data_dict)

    # Other info
    new_fem_data.settings = {
        'solution_type': 'STATIC',
        'output_res': 'NSTRAIN,ON\nNSTRESS,ON\n',
        'output_vis': 'NSTRAIN,ON\nNSTRESS,ON\n'}
    new_fem_data.element_groups = {'ALL': fem_data.elements.ids}
    new_fem_data.sections.update_data(
        'MAT_ALL', {'TYPE': 'SOLID', 'EGRP': 'ALL'})
    new_fem_data.constraints['spring'] = fem_data.constraints['spring']
    new_fem_data.material_overwritten = False

    new_fem_data.write('fistr', output_directory / basename)
    return


if __name__ == '__main__':
    main()

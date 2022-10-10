
import argparse
import multiprocessing as multi
import pathlib
import random

import femio
import numpy as np
import siml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'output_directory',
        type=pathlib.Path,
        help='Output directory path')
    parser.add_argument(
        '-n',
        '--n-train',
        type=int,
        default=10,
        help='The number of training data samples [10]')
    parser.add_argument(
        '-v',
        '--n-validation',
        type=int,
        default=10,
        help='The number of validation data samples [10]')
    parser.add_argument(
        '-j',
        '--min-n_element',
        type=int,
        default=10,
        help='The minimum number of elements [10]')
    parser.add_argument(
        '-k',
        '--max-n_element',
        type=int,
        default=20,
        help='The maximum number of elements [20]')
    parser.add_argument(
        '-p',
        '--max-process',
        type=int,
        default=None,
        help='If fed, set the maximum # of processes')
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=0,
        help='Random seed [0]')
    args = parser.parse_args()

    generator = DataGenerator(**vars(args))
    generator.generate()
    generator.preprocess()

    return


class DataGenerator:

    def __init__(
            self, output_directory, *,
            edge_length=.1,
            n_train=100, n_validation=10,
            seed=0,
            min_n_element=10, max_n_element=100,
            polynomial_degree=3, max_process=None):
        self.output_directory = output_directory / 'interim'
        self.edge_length = edge_length
        self.n_train = n_train
        self.n_validation = n_validation
        self.seed = seed
        self.min_n_element = min_n_element
        self.max_n_element = max_n_element
        self.max_process = siml.util.determine_max_process(max_process)

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        return

    def generate(self):
        """Create grid graph data.
        """

        self.mode = 'train'
        with multi.Pool(self.max_process) as pool:
            pool.map(self._generate_one_data, list(range(self.n_train)))

        self.mode = 'validation'
        with multi.Pool(self.max_process) as pool:
            pool.map(self._generate_one_data, list(range(self.n_validation)))

        return

    def preprocess(self):
        main_setting = siml.setting.MainSetting.read_settings_yaml(
            self.output_directory.parent / 'data.yml')
        preprocessor = siml.prepost.Preprocessor(
            main_setting, force_renew=True)
        preprocessor.preprocess_interim_data()

    def _generate_one_data(self, i_data):
        n_x_element = random.randint(
            self.min_n_element, self.max_n_element)
        n_y_element = random.randint(
            self.min_n_element, self.max_n_element)
        n_z_element = random.randint(
            self.min_n_element, self.max_n_element)

        fem_data = femio.generate_brick(
            'hex', n_x_element, n_y_element, n_z_element,
            x_length=self.edge_length*n_x_element,
            y_length=self.edge_length*n_y_element,
            z_length=self.edge_length*n_z_element)

        target_dict_data, fem_data = self.add_data(fem_data)
        dict_data = self.extract_feature(fem_data, target_dict_data)

        output_directory = self.output_directory / self.mode / str(i_data)
        self.save(output_directory, dict_data, fem_data)
        return

    def add_data(self, fem_data):
        nodes = fem_data.nodes.data
        cell_x = fem_data.convert_nodal2elemental(
            fem_data.nodes.data, calc_average=True)[:, 0]

        x_left = np.min(nodes[:, 0])
        x_right = np.max(nodes[:, 0])

        coeff = np.random.rand()
        phase = np.random.rand()
        init_phi = coeff * np.sin(
            2 * np.pi * cell_x / (x_right - x_left) + phase)[:, None]
        speed = np.random.rand()
        phi = np.stack([
            coeff * np.sin(
                2 * np.pi * cell_x
                / (x_right - x_left) + phase - speed * t)[:, None]
            for t in np.linspace(0., 1., 11)], axis=0)

        speed_vec = np.zeros((1, 3, 1))
        speed_vec[0, 0, 0] = speed
        cell_speed_vec = np.zeros((len(init_phi), 3, 1))
        cell_speed_vec[:, 0, 0] = speed
        dict_data = {
            'cell_initial_phi': init_phi,
            'cell_phi': phi,
            'cell_last_phi': phi[-1],
            'global_speed': speed_vec,
            'cell_speed': cell_speed_vec,
        }
        return dict_data, fem_data

    def extract_feature(self, fem_data, target_dict_data):

        facet_fem_data, signed_inc_facet2cell, facet_normal_vectors \
            = fem_data.calculate_normal_incidence_matrix()
        inc_cell2facet = facet_fem_data \
            .calculate_relative_incidence_metrix_element(
                fem_data, minimum_n_sharing=3)
        cell_volume = fem_data.calculate_element_volumes()
        facet_area = facet_fem_data.calculate_element_areas()

        facet_area_normal_vectors = np.einsum(
            'i,ik->ik', facet_area[..., 0], facet_normal_vectors)

        normal_inc = [
            signed_inc_facet2cell.multiply(
                facet_normal_vectors[:, [0]].T),
            signed_inc_facet2cell.multiply(
                facet_normal_vectors[:, [1]].T),
            signed_inc_facet2cell.multiply(
                facet_normal_vectors[:, [2]].T),
        ]  # (dim, n_cell, n_facet)-shape, where n_facet does not double count.
        area_normal_inc = [
            signed_inc_facet2cell.multiply(
                facet_area_normal_vectors[:, [0]].T),
            signed_inc_facet2cell.multiply(
                facet_area_normal_vectors[:, [1]].T),
            signed_inc_facet2cell.multiply(
                facet_area_normal_vectors[:, [2]].T),
        ]  # (dim, n_cell, n_facet)-shape, where n_facet does not double count.

        cell_adj = fem_data.calculate_adjacency_matrix_element()
        cell_nadj = siml.prepost.normalize_adjacency_matrix(cell_adj)

        cell_position = fem_data.convert_nodal2elemental(
            fem_data.nodes.data, calc_average=True)
        facet_position = facet_fem_data.convert_nodal2elemental(
            facet_fem_data.nodes.data, calc_average=True)
        facet_x = facet_position[:, 0]
        min_facet_x = np.min(facet_x)
        max_facet_x = np.max(facet_x)
        indices = np.arange(len(facet_x))
        min_indices = indices[np.abs(facet_x - min_facet_x) < 1e-5]
        max_indices = indices[np.abs(facet_x - max_facet_x) < 1e-5]
        periodic_flag = np.zeros((len(facet_x), 1))
        periodic_flag[max_indices] = -1
        periodic_flag[min_indices] = 1

        dict_data = {
            'signed_inc_cell2facet': signed_inc_facet2cell.T,
            'noraml_inc_facet2cell_x': normal_inc[0],
            'noraml_inc_facet2cell_y': normal_inc[1],
            'noraml_inc_facet2cell_z': normal_inc[2],
            'area_normal_inc_facet2cell_x': area_normal_inc[0],
            'area_normal_inc_facet2cell_y': area_normal_inc[1],
            'area_normal_inc_facet2cell_z': area_normal_inc[2],
            'inc_cell2facet': inc_cell2facet,
            'cell_volume': cell_volume,
            'facet_area': facet_area,
            'facet_normal': facet_normal_vectors,
            'facet_area_normal': facet_area_normal_vectors,
            'cell_adj': cell_adj,
            'cell_nadj': cell_nadj,
            'cell_position': cell_position,
            'facet_position': facet_position,
            'facet_periodic_flag': periodic_flag,
        }
        dict_data.update(target_dict_data)

        return dict_data

    def save(self, output_directory, dict_data, fem_data):
        siml.prepost.save_dict_data(output_directory, dict_data)
        fem_data_to_save = siml.prepost.update_fem_data(
            fem_data,
            {k: v for k, v in dict_data.items() if 'facet' not in k},
            allow_overwrite=True)
        fem_data_to_save.save(output_directory)
        fem_data_to_save.write('polyvtk', output_directory / 'mesh.vtu')
        (output_directory / 'converted').touch()
        return


if __name__ == '__main__':
    main()

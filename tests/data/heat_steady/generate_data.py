
import argparse
import multiprocessing as multi
import pathlib
import random

import femio
import numpy as np
import siml
from siml.preprocessing import converter
from siml.preprocessing import ScalingConverter
from siml.utils import fem_data_utils


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
        preprocessor = ScalingConverter(
            main_setting, force_renew=True
        )
        preprocessor.fit_transform()

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

        x_left = np.min(nodes[:, 0])
        x_right = np.max(nodes[:, 0])

        init_phi = np.zeros((len(nodes), 1))
        phi_left = np.random.rand()
        phi_right = np.random.rand()
        phi = (phi_right - phi_left) / (x_right - x_left) * (
            nodes[:, [0]] - x_left) + phi_left

        dirichlet_phi = np.ones(init_phi.shape) * np.nan
        x_min_filter = np.abs(
            nodes[:, [0]] - x_left) < self.edge_length * 1e-3
        x_max_filter = np.abs(
            nodes[:, [0]] - x_right) < self.edge_length * 1e-3
        init_phi[x_min_filter] = phi_left
        init_phi[x_max_filter] = phi_right
        dirichlet_phi[x_min_filter] = phi_left
        dirichlet_phi[x_max_filter] = phi_right

        dict_data = {
            'init_phi': init_phi,
            'phi': phi,
            'dirichlet_phi': dirichlet_phi,
        }
        return dict_data, fem_data

    def extract_feature(self, fem_data, target_dict_data):

        nodal_adj = fem_data.calculate_adjacency_matrix_node()
        nodal_nadj = siml.prepost.normalize_adjacency_matrix(nodal_adj)

        node = fem_data.nodal_data.get_attribute_data('node')

        nodal_x_grad_hop1, nodal_y_grad_hop1, nodal_z_grad_hop1 \
            = fem_data.calculate_spatial_gradient_adjacency_matrices(
                'nodal', n_hop=1, moment_matrix=True,
                normals=True, normal_weight=1.,
                consider_volume=False, adj=nodal_adj)

        dict_data = {
            'node': node,
            'nodal_adj': nodal_adj, 'nodal_nadj': nodal_nadj,
            'nodal_grad_x_1': nodal_x_grad_hop1,
            'nodal_grad_y_1': nodal_y_grad_hop1,
            'nodal_grad_z_1': nodal_z_grad_hop1,
        }
        dict_data.update(target_dict_data)

        return dict_data

    def save(self, output_directory, dict_data, fem_data):
        converter.save_dict_data(output_directory, dict_data)
        wrapped_data = fem_data_utils.FemDataWrapper(fem_data)
        wrapped_data.update_fem_data(dict_data, allow_overwrite=True)
        fem_data_to_save = wrapped_data.fem_data
        fem_data_to_save.save(output_directory)
        fem_data_to_save.write(
            'polyvtk', output_directory / 'mesh.vtu', overwrite=True)
        (output_directory / 'converted').touch()
        return


if __name__ == '__main__':
    main()

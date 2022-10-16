
from pathlib import Path
import glob
import unittest

import numpy as np


class EquivarianceBase(unittest.TestCase):

    def validate_results(
            self, original_results, transformed_results,
            *, rank0=None, rank1=None, rank2=None, validate_x=True, decimal=5,
            threshold_percent=1e-3):

        if rank0 is not None:
            scale = np.max(np.abs(original_results[0]['dict_y'][rank0]))

            for transformed_result in transformed_results:
                if validate_x:
                    np.testing.assert_almost_equal(
                        original_results[0]['dict_answer'][rank0],
                        transformed_result['dict_answer'][rank0],
                        decimal=decimal)

                print(
                    f"data_directory: {transformed_result['data_directory']}")
                self.print_vec(
                    original_results[0]['dict_y'][rank0] / scale,
                    'Transform x ML', 5)
                self.print_vec(
                    transformed_result['dict_y'][rank0] / scale,
                    'ML x Transform', 5)
                self.print_vec((
                    original_results[0]['dict_y'][rank0]
                    - transformed_result['dict_y'][rank0]) / scale,
                    'Diff', 5)
                self.compare_relative_rmse(
                    original_results[0]['dict_y'][rank0],
                    transformed_result['dict_y'][rank0],
                    threshold_percent=threshold_percent)
                np.testing.assert_almost_equal(
                    original_results[0]['dict_y'][rank0] / scale,
                    transformed_result['dict_y'][rank0] / scale,
                    decimal=decimal)

        if rank1 is not None:
            scale = np.max(np.abs(original_results[0]['dict_y'][rank1]))

            for transformed_result in transformed_results:
                print(
                    f"data_directory: {transformed_result['data_directory']}")
                orthogonal_matrix = self.load_orthogonal_matrix(
                    transformed_result['data_directory'])
                if validate_x:
                    np.testing.assert_almost_equal(
                        self.transform_rank1(
                            orthogonal_matrix,
                            original_results[0]['dict_x'][rank1]),
                        transformed_result['dict_x'][rank1], decimal=decimal)

                transformed_original = self.transform_rank1(
                    orthogonal_matrix, original_results[0]['dict_y'][rank1])
                self.print_vec(
                    transformed_original / scale, 'Transform x ML', 5)
                self.print_vec(
                    transformed_result['dict_y'][rank1] / scale,
                    'ML x Transform', 5)
                self.print_vec((
                    transformed_original
                    - transformed_result['dict_y'][rank1]) / scale,
                    'Diff', 5)
                self.compare_relative_rmse(
                    transformed_original, transformed_result['dict_y'][rank1],
                    threshold_percent=threshold_percent)
                np.testing.assert_almost_equal(
                    transformed_original / scale,
                    transformed_result['dict_y'][rank1] / scale,
                    decimal=decimal)

        if rank2 is not None:
            scale = np.max(np.abs(original_results[0]['dict_y'][rank2]))

            for transformed_result in transformed_results:
                print(
                    f"data_directory: {transformed_result['data_directory']}")
                orthogonal_matrix = self.load_orthogonal_matrix(
                    transformed_result['data_directory'])
                if validate_x:
                    np.testing.assert_almost_equal(
                        self.transform_rank2(
                            orthogonal_matrix,
                            original_results[0]['dict_answer'][rank2]),
                        transformed_result['dict_answer'][rank2],
                        decimal=decimal)

                transformed_original = self.transform_rank2(
                    orthogonal_matrix, original_results[0]['dict_y'][rank2])
                self.print_vec(
                    transformed_original / scale, 'Transform x ML', 5)
                self.print_vec(
                    transformed_result['dict_y'][rank2] / scale,
                    'ML x Transform', 5)
                self.print_vec((
                    transformed_original
                    - transformed_result['dict_y'][rank2]) / scale,
                    'Diff', 5)
                self.compare_relative_rmse(
                    transformed_original, transformed_result['dict_y'][rank2],
                    threshold_percent=threshold_percent)
                np.testing.assert_almost_equal(
                    transformed_original / scale,
                    transformed_result['dict_y'][rank2] / scale,
                    decimal=decimal)

        return

    def compare_relative_rmse(self, target, y, threshold_percent):
        target_scale = np.mean(target**2)**.5
        rmse = np.mean((y - target)**2)**.5
        self.assertLess(rmse / target_scale * 100, threshold_percent)

    def print_vec(self, x, name=None, n=None):
        if n is None:
            n = x.shape[0]
        print('--')
        if name is not None:
            print(name)
        if len(x.shape) == 4:
            for _x in x[:n, ..., 0]:
                print(_x)
        elif len(x.shape) == 3:
            for _x in x[:n, ..., 0]:
                print(_x)
        elif len(x.shape) == 2:
            for _x in x[:n]:
                print(_x)
        else:
            raise ValueError(f"Unexpected array shape: {x.shape}")
        print('--')
        return

    def transform_rank1(self, orthogonal_matrix, x):
        if len(x.shape) == 2:
            return np.array([orthogonal_matrix @ _x for _x in x])
        elif len(x.shape) == 3:
            n_feature = x.shape[-1]
            return np.stack([
                np.array([orthogonal_matrix @ _x for _x in x[..., i_feature]])
                for i_feature in range(n_feature)], axis=-1)
        else:
            raise ValueError(f"Unexpected x shape: {x.shape}")

    def identity(self, orthogonal_matrix, x):
        return x

    def transform_rank2(self, orthogonal_matrix, t):
        n_feature = t.shape[-1]
        return np.stack([
            np.array([
                orthogonal_matrix @ _t @ orthogonal_matrix.T
                for _t in t[..., i_feature]])
            for i_feature in range(n_feature)], axis=-1)

    def collect_transformed_paths(self, root_path, recursive=False):
        return [
            Path(g) for g in glob.glob(str(root_path), recursive=recursive)]

    def load_orthogonal_matrix(self, preprocessed_path):
        matrix_path = Path(
            str(preprocessed_path / 'orthogonal_matrix.txt').replace(
                'preprocessed', 'raw'))
        if matrix_path.is_file():
            return np.loadtxt(matrix_path)

        matrix_path = Path(
            str(preprocessed_path / 'orthogonal_matrix.txt').replace(
                'preprocessed', 'interim'))
        if matrix_path.is_file():
            return np.loadtxt(matrix_path)

        matrix_path = Path(
            str(preprocessed_path / 'rotation_matrix.npy').replace(
                'preprocessed', 'raw'))
        if matrix_path.is_file():
            return np.loadtxt(matrix_path)

        matrix_path = Path(
            str(preprocessed_path.parent / 'rotation_matrix.npy').replace(
                'preprocessed', 'raw'))
        if matrix_path.is_file():
            return np.loadtxt(matrix_path)

        matrix_path = Path(
            str(preprocessed_path / 'rotation_matrix.npy').replace(
                'preprocessed', 'interim'))
        if matrix_path.is_file():
            return np.loadtxt(matrix_path)

        matrix_path = Path(
            str(preprocessed_path.parent / 'rotation_matrix.npy').replace(
                'preprocessed', 'interim'))
        if matrix_path.is_file():
            return np.loadtxt(matrix_path)

        raise ValueError(
            f"Transformation matrix not found for: {preprocessed_path}")

    def evaluate_conservation(
            self, target, prediction, *,
            volume=None,
            target_time_series=False, prediction_time_series=False,
            decimal=7):

        if volume is None:
            if target_time_series:
                volume = np.ones(len(target[0]))[..., None]
            else:
                volume = np.ones(len(target))[..., None]

        if target_time_series:
            target_conservation = np.sum(target[0] * volume) / np.sum(volume)
        else:
            target_conservation = np.sum(target * volume) / np.sum(volume)

        if prediction_time_series:
            prediction_conservation = np.einsum(
                'ti...,i->t...', prediction, volume[..., 0]) / np.sum(volume)
        else:
            prediction_conservation = np.einsum(
                'i...,i->...', prediction, volume[..., 0]) / np.sum(volume)
        print(prediction_conservation - target_conservation)
        np.testing.assert_almost_equal(
            prediction_conservation - target_conservation, 0., decimal=decimal)
        return

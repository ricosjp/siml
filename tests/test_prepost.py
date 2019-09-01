from pathlib import Path
import shutil
import unittest

import numpy as np

import siml.prepost as pre
import siml.setting as setting


class TestPrepost(unittest.TestCase):

    def test_determine_output_directory(self):
        self.assertEqual(
            pre.determine_output_directory(
                Path('data/raw/a/b'), Path('test/sth'), 'raw'),
            Path('test/sth/a/b'))

    def test_normalize_adjacency_matrix(self):
        adj = np.array([
            [2., 1., 0.],
            [1., 10., 5.],
            [0., 5., 100.],
        ])
        nadj = pre.normalize_adjacency_matrix(adj)
        d_inv_sqrt = np.array([
            [3.**-.5, 0., 0.],
            [0., 16.**-.5, 0.],
            [0., 0., 105.**-.5],
        ])
        np.testing.assert_almost_equal(
            d_inv_sqrt @ adj @ d_inv_sqrt, nadj.toarray())

    def test_split_data_arrays(self):
        true_xs = [
            np.concatenate([
                np.stack([[0., 0.]] * 10000),
                np.stack([[1., 0.]] * 10000),
                np.stack([[0., 1.]] * 10000),
                np.stack([[1., 1.]] * 10000),
            ]),
            np.concatenate([
                np.stack([[0., 0.]] * 10000),
                np.stack([[1., 0.]] * 10000),
                np.stack([[0., 1.]] * 10000),
            ]),
        ]
        noised_xs = [
            np.concatenate([
                np.array([
                    [-.5, -.5],
                    [1.5, 1.5],
                ]),
                true_x + np.random.randn(*true_x.shape) * .1])
            for true_x in true_xs]
        fs = [noised_xs[0], noised_xs[1] / 2]
        ranges, list_split_data, centers, means, stds = pre.split_data_arrays(
            noised_xs, fs, n_split=3)

        array_means = np.transpose(np.stack(means), (1, 0, 2))
        array_stds = np.transpose(np.stack(stds), (1, 0, 2))
        answer = np.array([
                [0., 0.],
                [0., 1.],
                [1., 0.],
            ])
        np.testing.assert_array_almost_equal(centers, answer)
        np.testing.assert_array_almost_equal(
            array_means[0], answer, decimal=2)
        np.testing.assert_array_almost_equal(
            array_means[1], answer * .5, decimal=2)
        np.testing.assert_array_almost_equal(
            array_stds[0], np.ones(array_stds.shape[1:]) * .1, decimal=1)
        np.testing.assert_array_almost_equal(
            array_stds[1], np.ones(array_stds.shape[1:]) * .05, decimal=1)

    def test_preprocessor(self):
        data_setting = setting.DataSetting(
            interim='tests/data/prepost/interim',
            preprocessed='tests/data/prepost/preprocessed',
            pad=False
        )
        preprocess_setting = setting.PreprocessSetting(
            data_setting, {
                'identity': 'identity', 'std_scale': 'std_scale',
                'standardize': 'standardize'}
        )

        # Clean up data
        shutil.rmtree(data_setting.interim)
        shutil.rmtree(data_setting.preprocessed)
        data_setting.preprocessed.mkdir(parents=True)

        # Create data
        interim_paths = [
            data_setting.interim / 'a',
            data_setting.interim / 'b']
        for i, interim_path in enumerate(interim_paths):
            interim_path.mkdir(parents=True)
            n_element = np.random.randint(1e4)
            identity = np.random.randint(2, size=(n_element, 1))
            std_scale = np.random.rand(n_element, 3) * 5 * i
            standardize = np.random.randn(n_element, 5) * 2 * i \
                + i * np.array([[.1, .2, .3, .4, .5]])
            np.save(interim_path / 'identity.npy', identity)
            np.save(interim_path / 'std_scale.npy', std_scale)
            np.save(interim_path / 'standardize.npy', standardize)
            (interim_path / 'converted').touch()

        # Preprocess data
        preprocessor = pre.Preprocessor(preprocess_setting)
        preprocessor.preprocess_interim_data()

        # Test preprocessed data is as desired
        epsilon = 1e-5
        preprocessed_paths = [
            data_setting.preprocessed / 'a',
            data_setting.preprocessed / 'b']

        int_identity = np.concatenate([
            np.load(p / 'identity.npy') for p in interim_paths])
        pre_identity = np.concatenate([
            np.load(p / 'identity.npy') for p in preprocessed_paths])
        np.testing.assert_almost_equal(int_identity, pre_identity)

        int_std_scale = np.concatenate([
            np.load(p / 'std_scale.npy') for p in interim_paths])
        pre_std_scale = np.concatenate([
            np.load(p / 'std_scale.npy') for p in preprocessed_paths])
        np.testing.assert_almost_equal(
            int_std_scale / (np.std(int_std_scale, axis=0) + epsilon),
            pre_std_scale, decimal=5)
        np.testing.assert_almost_equal(
            np.std(pre_std_scale), 1. + epsilon, decimal=3)

        int_standardize = np.concatenate([
            np.load(p / 'standardize.npy') for p in interim_paths])
        pre_standardize = np.concatenate([
            np.load(p / 'standardize.npy') for p in preprocessed_paths])
        np.testing.assert_almost_equal(
            (int_standardize - np.mean(int_standardize, axis=0))
            / (np.std(int_standardize, axis=0) + epsilon),
            pre_standardize, decimal=5)
        np.testing.assert_almost_equal(
            np.std(pre_standardize, axis=0), 1. + epsilon, decimal=3)
        np.testing.assert_almost_equal(
            np.mean(pre_standardize, axis=0), np.zeros(5), decimal=5)

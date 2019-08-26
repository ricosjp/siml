from pathlib import Path
import unittest

import numpy as np

import siml.prepost as pre


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
            array_stds[0], np.ones(array_stds.shape[1:]) * .1, decimal=2)
        np.testing.assert_array_almost_equal(
            array_stds[1], np.ones(array_stds.shape[1:]) * .05, decimal=2)

from pathlib import Path
import unittest

import numpy as np

import siml.prepost as pre


class TestFemio(unittest.TestCase):

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

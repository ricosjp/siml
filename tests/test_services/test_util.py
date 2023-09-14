from pathlib import Path
import unittest

import numpy as np
import scipy.sparse as sp

import siml.util as util


class TestUtil(unittest.TestCase):

    def test_pad_array_ndarray(self):
        m = 5
        n = 7
        fs = (3, 4)
        cs = (2, 3)

        a = np.random.rand(m, *fs)
        w = np.random.rand(*fs, *cs)
        original_prod = np.einsum('mab,abcd->mcd', a, w)

        padded_a = util.pad_array(a, n)
        padded_prod = np.einsum('mab,abcd->mcd', padded_a, w)
        np.testing.assert_array_equal(padded_a.shape, [n] + list(fs))
        np.testing.assert_array_almost_equal(padded_prod[:m], original_prod)
        np.testing.assert_array_almost_equal(
            padded_prod[m:], np.zeros([n - m] + list(cs)))

    def test_pad_array_coomatrix(self):
        m = 5
        n = 7
        c = 3
        data = np.random.rand(m, m)

        a = sp.coo_matrix(data)
        w = np.random.rand(m, c)
        original_prod = a.dot(w)

        padded_a = util.pad_array(a, n)
        padded_w = util.pad_array(w, n)
        padded_prod = padded_a.dot(padded_w)
        np.testing.assert_array_equal(padded_a.shape, (n, n))
        np.testing.assert_array_almost_equal(padded_prod[:m], original_prod)
        np.testing.assert_array_almost_equal(
            padded_prod[m:], np.zeros((n - m, c)))

    def test_collect_data_directories(self):
        data_directories = util.collect_data_directories(
            Path('tests/data/deform/raw'),
            required_file_names=['*.msh', '*.cnt', '*.res.0.1'])
        self.assertEqual(len(data_directories), 10)

    def test_collect_data_directories_wildcard(self):
        data_directories = util.collect_data_directories(
            Path('tests/data/deform/raw/**/tet2_3*'),
            required_file_names=['*.msh', '*.cnt', '*.res.0.1'])
        self.assertEqual(len(data_directories), 5)


def test_get_top_level_directory():
    path = util.get_top_directory()

    # assumes that README exists if top-level directory
    assert (path / "README.md").exists()

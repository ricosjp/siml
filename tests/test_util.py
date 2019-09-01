from pathlib import Path
import unittest
import shutil

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

    def test_standardizer(self):
        n_data = 5
        dim = 3
        list_data = [
            np.random.randn(np.random.randint(2, 1e4), dim) * 2. * i + .5 * i
            for i in range(n_data)]
        out_directory = Path('tests/data/util_std')
        shutil.rmtree(out_directory, ignore_errors=True)
        data_files = [out_directory / f"data_{i}/x.npy" for i in range(n_data)]
        for data_file, d in zip(data_files, list_data):
            data_file.parent.mkdir(parents=True)
            np.save(data_file, d)

        all_data = np.concatenate(list_data)
        explicit_std = util.Standardizer(
            mean=np.mean(all_data, axis=0), std=np.std(all_data, axis=0))
        once_std = util.Standardizer.read_data(all_data)
        lazy_std = util.Standardizer.lazy_read_files(data_files)

        np.testing.assert_almost_equal(explicit_std.mean, once_std.mean)
        np.testing.assert_almost_equal(explicit_std.mean, lazy_std.mean)
        np.testing.assert_almost_equal(explicit_std.std, once_std.std)
        np.testing.assert_almost_equal(explicit_std.std, lazy_std.std)

        new_data = np.random.rand(100, dim)
        transformed_new_data = (new_data - explicit_std.mean) / (
            explicit_std.std + 1e-5)
        np.testing.assert_almost_equal(
            explicit_std.transform(new_data), transformed_new_data)
        np.testing.assert_almost_equal(
            explicit_std.inverse(transformed_new_data), new_data)

    def test_std_scale(self):
        n_data = 5
        dim = 3
        list_data = [
            np.random.randn(np.random.randint(2, 1e4), dim) * 2. * i + .5 * i
            for i in range(n_data)]
        out_directory = Path('tests/data/util_std')
        shutil.rmtree(out_directory, ignore_errors=True)
        data_files = [out_directory / f"data_{i}/x.npy" for i in range(n_data)]
        for data_file, d in zip(data_files, list_data):
            data_file.parent.mkdir(parents=True)
            np.save(data_file, d)

        all_data = np.concatenate(list_data)
        explicit_std = util.StandardScaler(
            mean=np.mean(all_data, axis=0), std=np.std(all_data, axis=0))
        once_std = util.StandardScaler.read_data(all_data)
        lazy_std = util.StandardScaler.lazy_read_files(data_files)

        np.testing.assert_almost_equal(explicit_std.std, once_std.std)
        np.testing.assert_almost_equal(explicit_std.std, lazy_std.std)

        new_data = np.random.rand(100, dim)
        transformed_new_data = new_data / (explicit_std.std + 1e-5)
        np.testing.assert_almost_equal(
            explicit_std.transform(new_data), transformed_new_data)
        np.testing.assert_almost_equal(
            explicit_std.inverse(transformed_new_data), new_data)

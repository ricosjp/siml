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
        all_data_file = out_directory / 'all_data.npy'
        np.save(all_data_file, all_data)
        once_std = util.PreprocessConverter(
            'standardize', data_files=[all_data_file])
        lazy_std = util.PreprocessConverter(
            'standardize', data_files=data_files)

        np.testing.assert_almost_equal(
            once_std.converter.mean_, lazy_std.converter.mean_)
        np.testing.assert_almost_equal(
            once_std.converter.var_, lazy_std.converter.var_)

        new_data = np.random.rand(100, dim)
        transformed_new_data = (new_data - np.mean(all_data, axis=0)) / np.std(
            all_data, axis=0)
        np.testing.assert_almost_equal(
            lazy_std.transform(new_data), transformed_new_data)
        np.testing.assert_almost_equal(
            lazy_std.inverse(transformed_new_data), new_data)

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
        all_data_file = out_directory / 'all_data.npy'
        np.save(all_data_file, all_data)
        once_std = util.PreprocessConverter(
            'std_scale', data_files=[all_data_file])
        lazy_std = util.PreprocessConverter(
            'std_scale', data_files=data_files)

        np.testing.assert_almost_equal(
            once_std.converter.var_, lazy_std.converter.var_)

        new_data = np.random.rand(100, dim)
        transformed_new_data = new_data / np.std(all_data, axis=0)
        np.testing.assert_almost_equal(
            lazy_std.transform(new_data), transformed_new_data)
        np.testing.assert_almost_equal(
            lazy_std.inverse(transformed_new_data), new_data)

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

    def test_max_abs_scaler(self):
        x = np.array([
            [1., 10., 100.],
            [2., -20., 200.],
            [3., 30., -300.],
        ])
        max_abs_scaler = util.MaxAbsScaler()
        max_abs_scaler.partial_fit(x)
        np.testing.assert_array_almost_equal(
            max_abs_scaler.max_, [3., 30., 300.])
        transformed_x = np.array([
            [1. / 3., 1. / 3., 1. / 3.],
            [2. / 3., -2. / 3., 2. / 3.],
            [3. / 3., 3. / 3., -3. / 3.],
        ])
        np.testing.assert_array_almost_equal(
            max_abs_scaler.transform(x),
            transformed_x)
        np.testing.assert_array_almost_equal(
            max_abs_scaler.inverse_transform(max_abs_scaler.transform(x)),
            x)

    def test_max_abs_scaler_sparse(self):
        x = sp.coo_matrix(np.array([
            [1.],
            [0.],
            [3.],
            [-4.],
            [0.],
        ]))
        max_abs_scaler = util.MaxAbsScaler()
        max_abs_scaler.partial_fit(x)
        np.testing.assert_array_almost_equal(
            max_abs_scaler.max_, [4.])

        transformed_x = np.array([
            [1. / 4.],
            [0. / 4.],
            [3. / 4.],
            [-4. / 4.],
            [0. / 4.],
        ])
        self.assertIsInstance(max_abs_scaler.transform(x), sp.coo_matrix)
        np.testing.assert_array_almost_equal(
            max_abs_scaler.transform(x).toarray(), transformed_x)
        np.testing.assert_array_almost_equal(
            max_abs_scaler.inverse_transform(
                max_abs_scaler.transform(x)).toarray(), x.toarray())

    def test_sparse_standard_scaler(self):
        x1 = sp.coo_matrix(np.array([
            [1.],
            [0.],
            [3.],
            [-4.],
            [0.],
            [1.],
            [0.],
            [3.],
            [-4.],
            [0.],
        ]))
        x2 = sp.coo_matrix(np.array([
            [0.],
            [1.],
            [.3],
            [0.],
            [0.],
            [1.],
            [.3],
            [0.],
        ]))
        answer_var = np.mean(np.concatenate([x1.toarray(), x2.toarray()])**2)
        sp_std_scaler = util.SparseStandardScaler()
        sp_std_scaler.partial_fit(x1)
        sp_std_scaler.partial_fit(x2)

        np.testing.assert_array_almost_equal(
            sp_std_scaler.var_, answer_var)

        self.assertIsInstance(sp_std_scaler.transform(x1), sp.coo_matrix)
        np.testing.assert_array_almost_equal(
            sp_std_scaler.transform(x1).toarray(),
            x1.toarray() / answer_var**.5)
        np.testing.assert_array_almost_equal(
            sp_std_scaler.inverse_transform(
                sp_std_scaler.transform(x1)).toarray(), x1.toarray())

    def test_sparse_standard_scaler_other_components(self):
        x1 = sp.coo_matrix(np.array([
            [1.],
            [0.],
            [3.],
            [-4.],
            [0.],
            [1.],
            [0.],
            [3.],
            [-4.],
            [0.],
        ]))
        x2 = sp.coo_matrix(np.array([
            [0.],
            [1.],
            [.3],
            [0.],
            [0.],
            [1.],
            [.3],
            [0.],
        ]))

        y1 = sp.coo_matrix(np.array([
            [1.],
            [0.],
            [13.],
            [-4.],
            [0.],
            [11.],
            [0.],
            [13.],
            [-14.],
            [0.],
        ]))
        y2 = sp.coo_matrix(np.array([
            [0.],
            [11.],
            [1.3],
            [0.],
            [0.],
            [11.],
            [1.3],
            [0.],
        ]))

        z1 = sp.coo_matrix(np.array([
            [21.],
            [0.],
            [3.],
            [-24.],
            [0.],
            [21.],
            [0.],
            [23.],
            [-24.],
            [0.],
        ]))
        z2 = sp.coo_matrix(np.array([
            [0.],
            [21.],
            [.3],
            [0.],
            [0.],
            [21.],
            [.3],
            [0.],
        ]))

        answer_var = np.mean(
            np.concatenate([x1.toarray(), x2.toarray()])**2
            + np.concatenate([y1.toarray(), y2.toarray()])**2
            + np.concatenate([z1.toarray(), z2.toarray()])**2
        )
        sp_std_scaler = util.SparseStandardScaler(other_components=['y', 'z'])
        sp_std_scaler.partial_fit(x1)
        sp_std_scaler.partial_fit(x2)
        sp_std_scaler.partial_fit(y1)
        sp_std_scaler.partial_fit(y2)
        sp_std_scaler.partial_fit(z1)
        sp_std_scaler.partial_fit(z2)

        np.testing.assert_array_almost_equal(
            sp_std_scaler.var_, answer_var)

        self.assertIsInstance(sp_std_scaler.transform(x1), sp.coo_matrix)
        np.testing.assert_array_almost_equal(
            sp_std_scaler.transform(x1).toarray(),
            x1.toarray() / answer_var**.5)
        np.testing.assert_array_almost_equal(
            sp_std_scaler.inverse_transform(
                sp_std_scaler.transform(x1)).toarray(), x1.toarray())

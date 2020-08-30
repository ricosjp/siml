from pathlib import Path
import pickle
import shutil
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

import siml.prepost as pre
import siml.setting as setting
import siml.trainer as trainer
import siml.util as util


def load_function(data_files, data_directory):
    # To be used in test_convert_raw_data_bypass_femio
    df = pd.read_csv(data_files[0], header=0, index_col=None)
    return {
        'a': np.reshape(df['a'].to_numpy(), (-1, 1)),
        'b': np.reshape(df['b'].to_numpy(), (-1, 1)),
        'c': np.reshape(df['c'].to_numpy(), (-1, 1))}, None


def conversion_function(fem_data, raw_directory=None):
    # To be used in test_preprocess_deform
    adj = fem_data.calculate_adjacency_matrix_element()
    nadj = pre.normalize_adjacency_matrix(adj)
    x_grad, y_grad, z_grad = \
        fem_data.calculate_spatial_gradient_adjacency_matrices('elemental')
    x_grad_2, y_grad_2, z_grad_2 = \
        fem_data.calculate_spatial_gradient_adjacency_matrices(
            'elemental', n_hop=2)
    global_modulus = np.mean(
        fem_data.elemental_data.get_attribute_data('modulus'), keepdims=True)

    tensor_stress = fem_data.convert_array2symmetric_matrix(
        fem_data.elemental_data.get_attribute_data('ElementalSTRESS'),
        from_engineering=False)[:, :, :, None]
    tensor_strain = fem_data.convert_array2symmetric_matrix(
        fem_data.elemental_data.get_attribute_data('ElementalSTRAIN'),
        from_engineering=True)[:, :, :, None]
    tensor_gauss_strain1 = fem_data.convert_array2symmetric_matrix(
        fem_data.elemental_data.get_attribute_data('GaussSTRAIN1'),
        from_engineering=True)[:, :, :, None]
    tensor_gauss_strain2 = fem_data.convert_array2symmetric_matrix(
        fem_data.elemental_data.get_attribute_data('GaussSTRAIN2'),
        from_engineering=True)[:, :, :, None]
    tensor_gauss_strain3 = fem_data.convert_array2symmetric_matrix(
        fem_data.elemental_data.get_attribute_data('GaussSTRAIN3'),
        from_engineering=True)[:, :, :, None]
    tensor_gauss_strain4 = fem_data.convert_array2symmetric_matrix(
        fem_data.elemental_data.get_attribute_data('GaussSTRAIN4'),
        from_engineering=True)[:, :, :, None]
    return {
        'adj': adj, 'nadj': nadj, 'global_modulus': global_modulus,
        'x_grad': x_grad, 'y_grad': y_grad, 'z_grad': z_grad,
        'x_grad_2': x_grad_2, 'y_grad_2': y_grad_2, 'z_grad_2': z_grad_2,
        'tensor_stress': tensor_stress, 'tensor_strain': tensor_strain,
        'tensor_gauss_strain1': tensor_gauss_strain1,
        'tensor_gauss_strain2': tensor_gauss_strain2,
        'tensor_gauss_strain3': tensor_gauss_strain3,
        'tensor_gauss_strain4': tensor_gauss_strain4,
    }


def conversion_function_heat_time_series(fem_data, raw_directory=None):
    # To be used in test_convert_heat_time_series
    temperature = fem_data.nodal_data.get_attribute_data('TEMPERATURE')
    return {'temperature': temperature}


def filter_function(fem_data, raw_directory=None, data_dict=None):
    # To be used in test_convert_raw_data_with_filter_function
    strain = fem_data.elemental_data.get_attribute_data('ElementalSTRAIN')
    return np.max(np.abs(strain)) < 1e2


class TestPrepost(unittest.TestCase):

    def test_determine_output_directory(self):
        self.assertEqual(
            pre.determine_output_directory(
                Path('data/raw/a/b'), Path('data/sth'), 'raw'),
            Path('data/sth/a/b'))
        self.assertEqual(
            pre.determine_output_directory(
                Path('tests/data/list/data/tet2_3_modulusx0.9000/interim'),
                Path('tests/data/list/preprocessed'), 'interim'),
            Path('tests/data/list/preprocessed/data/tet2_3_modulusx0.9000'))

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
        ranges, list_split_data, centers, means, stds, coverage \
            = pre.split_data_arrays(noised_xs, fs, n_split=3)

        array_means = np.transpose(np.stack(means), (1, 0, 2))
        array_stds = np.transpose(np.stack(stds), (1, 0, 2))
        answer = np.array([
            [0., 0.],
            [0., 1.],
            [1., 0.],
        ])
        np.testing.assert_array_almost_equal(centers, answer, decimal=1)
        np.testing.assert_array_almost_equal(
            array_means[0], answer, decimal=1)
        np.testing.assert_array_almost_equal(
            array_means[1], answer * .5, decimal=1)
        np.testing.assert_array_almost_equal(
            array_stds[0], np.ones(array_stds.shape[1:]) * .1, decimal=1)
        np.testing.assert_array_almost_equal(
            array_stds[1], np.ones(array_stds.shape[1:]) * .05, decimal=1)

    def test_convert_raw_data_bypass_femio(self):
        data_setting = setting.DataSetting(
            raw=Path('tests/data/csv_prepost/raw'),
            interim=Path('tests/data/csv_prepost/interim'))
        conversion_setting = setting.ConversionSetting(
            required_file_names=['*.csv'], skip_femio=True)

        main_setting = setting.MainSetting(
            data=data_setting, conversion=conversion_setting)

        shutil.rmtree(data_setting.interim_root, ignore_errors=True)
        shutil.rmtree(data_setting.preprocessed_root, ignore_errors=True)

        rc = pre.RawConverter(
            main_setting, recursive=True, load_function=load_function)
        rc.convert()

        interim_directory = data_setting.interim_root / 'train/1'
        expected_a = np.array([[1], [2], [3], [4]])
        expected_b = np.array([[2.1], [4.1], [6.1], [8.1]])
        expected_c = np.array([[3.2], [7.2], [8.2], [10.2]])
        np.testing.assert_almost_equal(
            np.load(interim_directory / 'a.npy'), expected_a)
        np.testing.assert_almost_equal(
            np.load(interim_directory / 'b.npy'), expected_b, decimal=5)
        np.testing.assert_almost_equal(
            np.load(interim_directory / 'c.npy'), expected_c, decimal=5)

    def test_preprocessor(self):
        data_setting = setting.DataSetting(
            interim=Path('tests/data/prepost/interim'),
            preprocessed=Path('tests/data/prepost/preprocessed'),
            pad=False
        )
        preprocess_setting = setting.PreprocessSetting(
            {
                'identity': 'identity', 'std_scale': 'std_scale',
                'standardize': 'standardize'}
        )
        main_setting = setting.MainSetting(
            preprocess=preprocess_setting.preprocess, data=data_setting,
            replace_preprocessed=False)
        main_setting.preprocess[  # pylint: disable=E1136
            'std_scale']['componentwise'] = True  # pylint: disable=E1136
        main_setting.preprocess[  # pylint: disable=E1136
            'standardize']['componentwise'] = True  # pylint: disable=E1136

        # Clean up data
        shutil.rmtree(data_setting.interim_root, ignore_errors=True)
        shutil.rmtree(data_setting.preprocessed_root, ignore_errors=True)
        data_setting.preprocessed_root.mkdir(parents=True)

        # Create data
        interim_paths = [
            data_setting.interim_root / 'a',
            data_setting.interim_root / 'b']
        for i, interim_path in enumerate(interim_paths):
            interim_path.mkdir(parents=True)
            n_element = int(1e5)
            identity = np.random.randint(2, size=(n_element, 1))
            std_scale = np.random.rand(n_element, 3) * 5 * i
            standardize = np.random.randn(n_element, 5) * 2 * i \
                + i * np.array([[.1, .2, .3, .4, .5]])
            np.save(interim_path / 'identity.npy', identity)
            np.save(interim_path / 'std_scale.npy', std_scale)
            np.save(interim_path / 'standardize.npy', standardize)
            (interim_path / 'converted').touch()

        # Preprocess data
        preprocessor = pre.Preprocessor(main_setting)
        preprocessor.preprocess_interim_data()

        # Test preprocessed data is as desired
        epsilon = 1e-5
        preprocessed_paths = [
            data_setting.preprocessed_root / 'a',
            data_setting.preprocessed_root / 'b']

        int_identity = np.concatenate([
            np.load(p / 'identity.npy') for p in interim_paths])
        pre_identity = np.concatenate([
            np.load(p / 'identity.npy') for p in preprocessed_paths])

        np.testing.assert_almost_equal(
            int_identity, pre_identity, decimal=3)

        int_std_scale = np.concatenate([
            np.load(p / 'std_scale.npy') for p in interim_paths])
        pre_std_scale = np.concatenate([
            np.load(p / 'std_scale.npy') for p in preprocessed_paths])

        np.testing.assert_almost_equal(
            int_std_scale / (np.std(int_std_scale, axis=0) + epsilon),
            pre_std_scale, decimal=3)
        np.testing.assert_almost_equal(
            np.std(pre_std_scale), 1. + epsilon, decimal=3)

        int_standardize = np.concatenate([
            np.load(p / 'standardize.npy') for p in interim_paths])
        pre_standardize = np.concatenate([
            np.load(p / 'standardize.npy') for p in preprocessed_paths])

        np.testing.assert_almost_equal(
            (int_standardize - np.mean(int_standardize, axis=0))
            / (np.std(int_standardize, axis=0) + epsilon),
            pre_standardize, decimal=3)
        np.testing.assert_almost_equal(
            np.std(pre_standardize, axis=0), 1. + epsilon, decimal=3)
        np.testing.assert_almost_equal(
            np.mean(pre_standardize, axis=0), np.zeros(5), decimal=3)

    def test_postprocessor(self):
        data_setting = setting.DataSetting(
            interim=Path('tests/data/prepost/interim'),
            preprocessed=Path('tests/data/prepost/preprocessed'),
            pad=False
        )
        preprocess_setting = setting.PreprocessSetting(
            {
                'identity': 'identity', 'std_scale': 'std_scale',
                'standardize': 'standardize'}
        )
        main_setting = setting.MainSetting(
            preprocess=preprocess_setting.preprocess, data=data_setting,
            replace_preprocessed=False)

        # Clean up data
        shutil.rmtree(data_setting.interim_root, ignore_errors=True)
        shutil.rmtree(data_setting.preprocessed_root, ignore_errors=True)
        data_setting.preprocessed_root.mkdir(parents=True)

        # Create data
        interim_paths = [
            data_setting.interim_root / 'a',
            data_setting.interim_root / 'b']
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
        preprocessor = pre.Preprocessor(main_setting)
        preprocessor.preprocess_interim_data()

        postprocessor = pre.Converter(
            data_setting.preprocessed_root / 'preprocessors.pkl')
        preprocessed_paths = [
            data_setting.preprocessed_root / 'a',
            data_setting.preprocessed_root / 'b']
        for interim_path, preprocessed_path in zip(
                interim_paths, preprocessed_paths):
            dict_data_x = {
                'identity': np.load(preprocessed_path / 'identity.npy'),
                'std_scale': np.load(preprocessed_path / 'std_scale.npy')}
            dict_data_y = {
                'standardize': np.load(preprocessed_path / 'standardize.npy')}
            inv_dict_data_x, inv_dict_data_y = postprocessor.postprocess(
                dict_data_x, dict_data_y)
            for k, v in inv_dict_data_x.items():
                interim_data = np.load(interim_path / (k + '.npy'))
                np.testing.assert_almost_equal(interim_data, v, decimal=5)
            for k, v in inv_dict_data_y.items():
                interim_data = np.load(interim_path / (k + '.npy'))
                np.testing.assert_almost_equal(interim_data, v, decimal=5)

    def test_preprocess_deform(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/data.yml'))
        main_setting.data.interim = [Path(
            'tests/data/deform/test_prepost/interim')]
        main_setting.data.preprocessed = [Path(
            'tests/data/deform/test_prepost/preprocessed')]

        shutil.rmtree(main_setting.data.interim_root, ignore_errors=True)
        shutil.rmtree(main_setting.data.preprocessed_root, ignore_errors=True)

        raw_converter = pre.RawConverter(
            main_setting, conversion_function=conversion_function)
        raw_converter.convert()
        p = pre.Preprocessor(main_setting)
        p.preprocess_interim_data()

        interim_strain = np.load(
            'tests/data/deform/test_prepost/interim/train/'
            'tet2_3_modulusx1.0000/elemental_strain.npy')
        preprocessed_strain = np.load(
            'tests/data/deform/test_prepost/preprocessed/train/'
            'tet2_3_modulusx1.0000/elemental_strain.npy')
        ratio_strain = interim_strain / preprocessed_strain
        np.testing.assert_almost_equal(
            ratio_strain - np.mean(ratio_strain), 0.)

        interim_y_grad = sp.load_npz(
            'tests/data/deform/test_prepost/interim/train/'
            'tet2_3_modulusx1.0000/y_grad.npz')
        preprocessed_y_grad = sp.load_npz(
            'tests/data/deform/test_prepost/preprocessed/train/'
            'tet2_3_modulusx1.0000/y_grad.npz')

        ratio_y_grad = interim_y_grad.toarray() \
            / preprocessed_y_grad.toarray()
        np.testing.assert_almost_equal(
            ratio_y_grad - np.mean(ratio_y_grad), 0.)

    def test_convert_raw_data_with_filter_function(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/test_prepost_to_filter/data.yml'))
        shutil.rmtree(main_setting.data.interim_root, ignore_errors=True)

        raw_converter = pre.RawConverter(
            main_setting, filter_function=filter_function)
        raw_converter.convert()

        actual_directories = sorted(util.collect_data_directories(
            main_setting.data.interim,
            required_file_names=['elemental_strain.npy']))
        expected_directories = sorted([
            main_setting.data.interim_root / 'tet2_3_modulusx0.9000',
            main_setting.data.interim_root / 'tet2_3_modulusx1.1000',
            main_setting.data.interim_root / 'tet2_4_modulusx1.0000',
            main_setting.data.interim_root / 'tet2_4_modulusx1.1000'])
        np.testing.assert_array_equal(actual_directories, expected_directories)

    def test_generate_converters(self):
        preprocessors_file = Path('tests/data/prepost/preprocessors.pkl')
        real_file_converter = pre.Converter(preprocessors_file)
        with open(preprocessors_file, 'rb') as f:
            file_like_object_converter = pre.Converter(f)
        np.testing.assert_almost_equal(
            real_file_converter.converters['standardize'].converter.var_,
            file_like_object_converter.converters[
                'standardize'].converter.var_)

    def test_concatenate_preprocessed_data(self):
        preprocessed_base_directory = Path(
            'tests/data/linear/preprocessed/train')
        concatenated_directory = Path('tests/data/linear/concatenated')
        shutil.rmtree(concatenated_directory, ignore_errors=True)

        pre.concatenate_preprocessed_data(
            preprocessed_base_directory, concatenated_directory,
            variable_names=['x1', 'x2', 'y'], ratios=(1., 0., 0.))

        for name in ['x1', 'x2', 'y']:
            actual = np.load(concatenated_directory / f"train/{name}.npy")
            answer = np.concatenate([
                np.load(preprocessed_base_directory / f"0/{name}.npy"),
                np.load(preprocessed_base_directory / f"1/{name}.npy")])
            np.testing.assert_almost_equal(
                np.max(actual), np.max(answer), decimal=5)
            np.testing.assert_almost_equal(
                np.min(actual), np.min(answer), decimal=5)
            np.testing.assert_almost_equal(
                np.std(actual), np.std(answer), decimal=5)
            np.testing.assert_almost_equal(
                np.mean(actual), np.mean(answer), decimal=5)

    def test_train_concatenated_data(self):
        preprocessed_base_directory = Path(
            'tests/data/linear/preprocessed/train')
        concatenated_directory = Path('tests/data/linear/concatenated')
        shutil.rmtree(concatenated_directory, ignore_errors=True)

        pre.concatenate_preprocessed_data(
            preprocessed_base_directory, concatenated_directory,
            variable_names=['x1', 'x2', 'y'], ratios=(.9, 0.1, 0.))

        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_concatenated.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1e-5)

    def test_preprocess_timeseries_data(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/csv_timeseries/lstm.yml'))

        shutil.rmtree(main_setting.data.preprocessed_root, ignore_errors=True)

        p = pre.Preprocessor(main_setting)
        p.preprocess_interim_data()

        c = pre.Converter(
            main_setting.data.preprocessed_root / 'preprocessors.pkl')
        original_dict_x = {
            'a': np.load(
                main_setting.data.interim_root / 'train/0/a.npy')[:, None]}
        preprocessed_dict_x = c.preprocess(original_dict_x)
        postprocessed_dict_x, _ = c.postprocess(preprocessed_dict_x, {})
        np.testing.assert_almost_equal(
            preprocessed_dict_x['a'],
            np.load(
                main_setting.data.preprocessed_root
                / 'train/0/a.npy')[:, None])
        np.testing.assert_almost_equal(
            original_dict_x['a'][:, 0], postprocessed_dict_x['a'])

    def test_preprocess_same_as(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/ode/data.yml'))

        shutil.rmtree(main_setting.data.preprocessed_root, ignore_errors=True)
        preprocessor = pre.Preprocessor(main_setting, force_renew=True)
        preprocessor.preprocess_interim_data()
        data_directory = main_setting.data.preprocessed_root / 'train/0'
        y0 = np.load(data_directory / 'y0.npy')
        y0_initial = np.load(data_directory / 'y0_initial.npy')
        np.testing.assert_almost_equal(y0[0], y0_initial[0])
        np.testing.assert_almost_equal(y0_initial - y0_initial[0], 0.)

    def test_preprocess_power(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/power.yml'))

        shutil.rmtree(main_setting.data.preprocessed_root, ignore_errors=True)
        preprocessor = pre.Preprocessor(main_setting, force_renew=True)
        preprocessor.preprocess_interim_data()
        data_directory = main_setting.data.preprocessed_root \
            / 'train/tet2_3_modulusx0.9000'
        preprocessed_x_grad = sp.load_npz(
            data_directory / 'x_grad.npz')
        reference_x_grad = sp.load_npz(
            'tests/data/deform/interim/train/tet2_3_modulusx0.9000'
            '/x_grad.npz').toarray()
        with open(
                main_setting.data.preprocessed_root / 'preprocessors.pkl',
                'rb') as f:
            preprocess_converter = pickle.load(f)['x_grad'][
                'preprocess_converter']
        std = preprocess_converter.converter.std_
        np.testing.assert_almost_equal(
            preprocessed_x_grad.toarray() * std**.5, reference_x_grad)
        np.testing.assert_almost_equal(
            preprocess_converter.converter.inverse_transform(
                preprocessed_x_grad).toarray(), reference_x_grad)

    def test_convert_heat_time_series(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/heat_time_series/data.yml'))

        shutil.rmtree(main_setting.data.interim_root, ignore_errors=True)

        rc = pre.RawConverter(
            main_setting, recursive=True, write_ucd=False,
            conversion_function=conversion_function_heat_time_series)
        rc.convert()

    def test_preprocess_interim_list(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/list/data.yml'))
        shutil.rmtree(main_setting.data.preprocessed_root, ignore_errors=True)

        preprocessor = pre.Preprocessor(main_setting)
        preprocessor.preprocess_interim_data()
        self.assertTrue(Path(
            'tests/data/list/preprocessed/data/tet2_3_modulusx0.9500'
        ).exists())
        self.assertTrue(Path(
            'tests/data/list/preprocessed/data/tet2_4_modulusx0.9000'
        ).exists())

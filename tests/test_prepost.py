from pathlib import Path
import shutil
import unittest

import numpy as np

import siml.prepost as pre
import siml.setting as setting
import siml.trainer as trainer


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
        np.testing.assert_array_almost_equal(centers, answer, decimal=1)
        np.testing.assert_array_almost_equal(
            array_means[0], answer, decimal=1)
        np.testing.assert_array_almost_equal(
            array_means[1], answer * .5, decimal=1)
        np.testing.assert_array_almost_equal(
            array_stds[0], np.ones(array_stds.shape[1:]) * .1, decimal=1)
        np.testing.assert_array_almost_equal(
            array_stds[1], np.ones(array_stds.shape[1:]) * .05, decimal=1)

    def test_preprocessor(self):
        data_setting = setting.DataSetting(
            interim=Path('tests/data/prepost/interim'),
            preprocessed=Path('tests/data/prepost/preprocessed'),
            pad=False
        )
        preprocess_setting = setting.PreprocessSetting(
            data_setting, {
                'identity': 'identity', 'std_scale': 'std_scale',
                'standardize': 'standardize'}
        )

        # Clean up data
        shutil.rmtree(data_setting.interim, ignore_errors=True)
        shutil.rmtree(data_setting.preprocessed, ignore_errors=True)
        data_setting.preprocessed.mkdir(parents=True)

        # Create data
        interim_paths = [
            data_setting.interim / 'a',
            data_setting.interim / 'b']
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
            data_setting, {
                'identity': 'identity', 'std_scale': 'std_scale',
                'standardize': 'standardize'}
        )

        # Clean up data
        shutil.rmtree(data_setting.interim, ignore_errors=True)
        shutil.rmtree(data_setting.preprocessed, ignore_errors=True)
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

        postprocessor = pre.Converter(
            Path('tests/data/prepost/preprocessed/preprocessors.pkl'))
        preprocessed_paths = [
            data_setting.preprocessed / 'a',
            data_setting.preprocessed / 'b']
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
        interim = Path('tests/data/deform/test_prepost/interim')
        preprocessed = Path('tests/data/deform/test_prepost/preprocessed')
        shutil.rmtree(interim, ignore_errors=True)
        shutil.rmtree(preprocessed, ignore_errors=True)

        def conversion_function(fem_data, raw_directory=None):
            adj, _ = fem_data.calculate_adjacency_matrix_element()
            nadj = pre.normalize_adjacency_matrix(adj)
            return {'adj': adj, 'nadj': nadj}
        pre.convert_raw_data(
            Path('tests/data/deform/raw'),
            ['elemental_strain', 'elemental_stress',
             'modulus', 'poisson_ratio'],
            output_base_directory=interim,
            recursive=True, conversion_function=conversion_function)
        p = pre.Preprocessor.read_settings('tests/data/deform/data.yml')
        p.setting.data.interim = interim
        p.setting.data.preprocessed = preprocessed
        p.preprocess_interim_data()

    def test_generate_converters(self):
        preprocessors_file = Path('tests/data/prepost/preprocessors.pkl')
        real_file_converter = pre.Converter(preprocessors_file)
        with open(preprocessors_file, 'rb') as f:
            file_like_object_converter = pre.Converter(f)
        np.testing.assert_almost_equal(
            real_file_converter.converters['sharp20'].parameters['std'],
            file_like_object_converter.converters['sharp20'].parameters['std'])

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
            np.testing.assert_almost_equal(np.max(actual), np.max(answer))
            np.testing.assert_almost_equal(np.min(actual), np.min(answer))
            np.testing.assert_almost_equal(np.std(actual), np.std(answer))
            np.testing.assert_almost_equal(np.mean(actual), np.mean(answer))

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

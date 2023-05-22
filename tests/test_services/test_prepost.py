from pathlib import Path
import pickle
import shutil
import unittest

import numpy as np

import siml.prepost as pre
import siml.setting as setting
import siml.trainer as trainer
from siml.preprocessing import ScalingConverter, ScalersComposition


class TestPrepost(unittest.TestCase):

    def test_preprocessors_pkl_eliminates_sklearn_objects(self):
        with open(
                'tests/data/deform/preprocessed/preprocessors.pkl', 'rb') as f:
            dict_data = pickle.load(f)
        for key, value in dict_data.items():
            if key == "variable_name_to_scalers":
                continue
            self.assertTrue(isinstance(value['preprocess_converter'], dict))

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

    def test_normalize_adjacency_matrix_wo_diag(self):
        adj = np.array([
            [0., 5., 0.],
            [1., 0., 1.],
            [0., 0., 0.],
        ])
        nadj = pre.normalize_adjacency_matrix(adj)
        d_inv_sqrt = np.array([
            [6.**-.5, 0., 0.],
            [0., 3.**-.5, 0.],
            [0., 0., 1.**-.5],
        ])
        np.testing.assert_almost_equal(
            d_inv_sqrt @ (adj + np.eye(3)) @ d_inv_sqrt, nadj.toarray())

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
        preprocessor = ScalingConverter(main_setting)
        preprocessor.fit_transform()

        postprocessor = ScalersComposition.create_from_file(
            converter_parameters_pkl=(
                data_setting.preprocessed_root / 'preprocessors.pkl'
            )
        )
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
            inv_dict_data_x = postprocessor.inverse_transform_dict(dict_data_x)
            inv_dict_data_y = postprocessor.inverse_transform_dict(dict_data_y)
            for k, v in inv_dict_data_x.items():
                interim_data = np.load(interim_path / (k + '.npy'))
                np.testing.assert_almost_equal(interim_data, v, decimal=5)
            for k, v in inv_dict_data_y.items():
                interim_data = np.load(interim_path / (k + '.npy'))
                np.testing.assert_almost_equal(interim_data, v, decimal=5)

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
                np.load(preprocessed_base_directory / f"1/{name}.npy"),
                np.load(preprocessed_base_directory / f"2/{name}.npy"),
                np.load(preprocessed_base_directory / f"3/{name}.npy"),
                np.load(preprocessed_base_directory / f"4/{name}.npy"),
            ])
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

        p = ScalingConverter(main_setting)
        p.fit_transform()

        converter = ScalersComposition.create_from_file(
            converter_parameters_pkl=(
                main_setting.data.preprocessed_root / 'preprocessors.pkl'
            )
        )
        original_dict_x = {
            'a': np.load(
                main_setting.data.interim_root / 'train/0/a.npy')}
        preprocessed_dict_x = converter.transform_dict(original_dict_x)
        postprocessed_dict_x = converter.inverse_transform_dict(
            preprocessed_dict_x
        )
        np.testing.assert_almost_equal(
            preprocessed_dict_x['a'],
            np.load(
                main_setting.data.preprocessed_root
                / 'train/0/a.npy'))
        np.testing.assert_almost_equal(
            original_dict_x['a'], postprocessed_dict_x['a'])

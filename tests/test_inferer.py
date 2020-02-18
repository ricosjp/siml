from pathlib import Path
import shutil
import unittest

import femio
import numpy as np

import siml.prepost as prepost
import siml.setting as setting
import siml.inferer as inferer


class TestInferer(unittest.TestCase):

    def test_infer_with_preprocessed_data(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/pretrained/settings.yaml'))
        ir = inferer.Inferer(main_setting)
        if ir.setting.trainer.output_directory.exists():
            shutil.rmtree(ir.setting.trainer.output_directory)
        res = ir.infer(
            model_path=Path('tests/data/linear/pretrained'),
            preprocessed_data_directory=Path(
                'tests/data/linear/preprocessed/validation'),
            converter_parameters_pkl=Path(
                'tests/data/linear/preprocessed/preprocessors.pkl'))
        np.testing.assert_almost_equal(
            res[0]['dict_y']['y'],
            np.load('tests/data/linear/interim/validation/0/y.npy'), decimal=3)
        np.testing.assert_array_less(res[0]['loss'], 1e-7)

    def test_infer_with_raw_data(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/pretrained/settings.yaml'))
        ir = inferer.Inferer(main_setting)
        if ir.setting.trainer.output_directory.exists():
            shutil.rmtree(ir.setting.trainer.output_directory)

        def conversion_function(fem_data, raw_directory=None):
            adj, _ = fem_data.calculate_adjacency_matrix_element()
            nadj = prepost.normalize_adjacency_matrix(adj)
            return {'adj': adj, 'nadj': nadj}

        res_from_raw = ir.infer(
            model_path=Path('tests/data/deform/pretrained'),
            raw_data_directory=Path(
                'tests/data/deform/raw/test/tet2_4_modulusx0.9500'),
            converter_parameters_pkl=Path(
                'tests/data/deform/preprocessed/preprocessors.pkl'),
            conversion_function=conversion_function, save=False)

        res_from_preprocessed = ir.infer(
            model_path=Path('tests/data/deform/pretrained'),
            preprocessed_data_directory=Path(
                'tests/data/deform/preprocessed/test/'
                'tet2_4_modulusx0.9500'),
            converter_parameters_pkl=Path(
                'tests/data/deform/preprocessed/preprocessors.pkl'))

        np.testing.assert_almost_equal(
            res_from_raw[0]['dict_y']['elemental_stress'],
            res_from_preprocessed[0]['dict_y']['elemental_stress'],
            decimal=3)
        np.testing.assert_almost_equal(
            res_from_raw[0]['loss'], res_from_preprocessed[0]['loss'])
        np.testing.assert_array_less(res_from_raw[0]['loss'], 1e-2)

    def test_infer_with_raw_data_wo_answer(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/pretrained/settings.yaml'))
        ir = inferer.Inferer(main_setting)
        if ir.setting.trainer.output_directory.exists():
            shutil.rmtree(ir.setting.trainer.output_directory)

        def conversion_function(fem_data, raw_directory=None):
            adj, _ = fem_data.calculate_adjacency_matrix_element()
            nadj = prepost.normalize_adjacency_matrix(adj)
            return {'adj': adj, 'nadj': nadj}

        res_from_raw = ir.infer(
            model_path=Path('tests/data/deform/pretrained'),
            raw_data_directory=Path(
                'tests/data/deform/external/tet2_4_modulusx0.9500'),
            converter_parameters_pkl=Path(
                'tests/data/deform/preprocessed/preprocessors.pkl'),
            conversion_function=conversion_function, save=False)
        res_from_preprocessed = ir.infer(
            model_path=Path('tests/data/deform/pretrained'),
            preprocessed_data_directory=Path(
                'tests/data/deform/preprocessed/test/'
                'tet2_4_modulusx0.9500'),
            converter_parameters_pkl=Path(
                'tests/data/deform/preprocessed/preprocessors.pkl'))
        np.testing.assert_almost_equal(
            res_from_raw[0]['dict_y']['elemental_stress'],
            res_from_preprocessed[0]['dict_y']['elemental_stress'],
            decimal=3)

    def test_infer_with_raw_data_wo_answer_with_model_file(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/incomplete_pretrained/settings.yaml'))
        ir = inferer.Inferer(main_setting)
        if ir.setting.trainer.output_directory.exists():
            shutil.rmtree(ir.setting.trainer.output_directory)

        def conversion_function(fem_data, raw_directory=None):
            adj, _ = fem_data.calculate_adjacency_matrix_element()
            nadj = prepost.normalize_adjacency_matrix(adj)
            return {'adj': adj, 'nadj': nadj}

        res_from_raw = ir.infer(
            model_path=Path(
                'tests/data/deform/incomplete_pretrained/'
                'snapshot_epoch_5000.pth'),
            raw_data_directory=Path(
                'tests/data/deform/external/tet2_4_modulusx0.9500'),
            converter_parameters_pkl=Path(
                'tests/data/deform/preprocessed/preprocessors.pkl'),
            conversion_function=conversion_function, save=False)
        res_from_preprocessed = ir.infer(
            model_path=Path('tests/data/deform/pretrained'),
            preprocessed_data_directory=Path(
                'tests/data/deform/preprocessed/test/'
                'tet2_4_modulusx0.9500'),
            converter_parameters_pkl=Path(
                'tests/data/deform/preprocessed/preprocessors.pkl'))
        np.testing.assert_almost_equal(
            res_from_raw[0]['dict_y']['elemental_stress'],
            res_from_preprocessed[0]['dict_y']['elemental_stress'],
            decimal=3)

    def test_infer_to_write_simulation_file(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/incomplete_pretrained/settings.yaml'))
        output_directory = Path('tests/data/deform/write_simulation')

        ir = inferer.Inferer(main_setting)
        if output_directory.exists():
            shutil.rmtree(output_directory)

        res_from_preprocessed = ir.infer(
            model_path=Path('tests/data/deform/pretrained'),
            output_directory=output_directory,
            preprocessed_data_directory=Path(
                'tests/data/deform/preprocessed/test/'
                'tet2_4_modulusx0.9500'),
            converter_parameters_pkl=Path(
                'tests/data/deform/preprocessed/preprocessors.pkl'),
            write_simulation_base=Path(
                'tests/data/deform/raw/test/tet2_4_modulusx0.9500'),
            write_simulation=True, write_simulation_type='ucd')
        fem_data = femio.FEMData.read_files(
            'ucd', [output_directory / 'mesh.inp'])
        np.testing.assert_almost_equal(
            fem_data.access_attribute('elemental_stress'),
            res_from_preprocessed[0]['dict_y']['elemental_stress'],
            decimal=7)

    def test_infer_simplified_model(self):
        setting_yaml = Path('tests/data/simplified/mlp.yml')
        model_file = Path(
            'tests/data/simplified/pretrained/snapshot_epoch_1000.pth')
        converter_parameters_pkl = Path(
            'tests/data/simplified/pretrained/preprocessors.pkl')
        ir = inferer.Inferer.read_settings(setting_yaml)
        seed_a = np.random.rand(10, 1)
        raw_dict_x = {
            'a': np.concatenate([seed_a, seed_a * 2, seed_a * 3], axis=1),
            'b': np.random.rand(10, 1) * 100.}

        answer_raw_dict_y = {'c': raw_dict_x['a'] * raw_dict_x['b']}
        inversed_dict_y, loss = ir.infer_simplified_model(
            model_file, raw_dict_x, answer_raw_dict_y=answer_raw_dict_y,
            converter_parameters_pkl=converter_parameters_pkl)
        rmse = np.mean((inversed_dict_y['c'] - answer_raw_dict_y['c'])**2)**.5
        self.assertLess(rmse, 5.)
        self.assertLess(loss, 2e-3)

    def test_infer_timeseries(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform_timeseries/pretrained/settings.yaml'))
        ir = inferer.Inferer(main_setting)
        if ir.setting.trainer.output_directory.exists():
            shutil.rmtree(ir.setting.trainer.output_directory)
        preprocessed_data_directory = Path(
            'tests/data/deform_timeseries/preprocessed/train'
            '/tet2_3_modulusx1.0000')
        res = ir.infer(
            model_path=Path('tests/data/deform_timeseries/pretrained'),
            preprocessed_data_directory=preprocessed_data_directory,
            converter_parameters_pkl=Path(
                'tests/data/deform_timeseries/preprocessed/preprocessors.pkl'))
        np.testing.assert_almost_equal(
            res[0]['dict_y']['stress'] * 1e-5,
            np.load(
                'tests/data/deform_timeseries/interim/train'
                '/tet2_3_modulusx1.0000/stress.npy') * 1e-5,
            decimal=3)
        np.testing.assert_array_less(res[0]['loss'], 1e-3)

    def test_infer_res_gcn(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/pretrained_res_gcn/settings.yml'))
        ir = inferer.Inferer(main_setting)
        if ir.setting.trainer.output_directory.exists():
            shutil.rmtree(ir.setting.trainer.output_directory)
        preprocessed_data_directory = Path(
            'tests/data/deform/preprocessed/train/tet2_3_modulusx1.0000')
        res = ir.infer(
            model_path=Path('tests/data/deform/pretrained_res_gcn'),
            preprocessed_data_directory=preprocessed_data_directory,
            converter_parameters_pkl=Path(
                'tests/data/deform/preprocessed/preprocessors.pkl'))
        np.testing.assert_almost_equal(
            res[0]['dict_y']['elemental_stress'] * 1e-5,
            np.load(
                'tests/data/deform/interim/train'
                '/tet2_3_modulusx1.0000/elemental_stress.npy') * 1e-5,
            decimal=3)
        np.testing.assert_array_less(res[0]['loss'], 1e-3)

from pathlib import Path
import shutil
import unittest

from chainer import testing
import numpy as np

import siml.femio as femio
import siml.prepost as prepost
import siml.setting as setting
import siml.trainer as trainer


class TestTrainer(unittest.TestCase):

    def test_train_cpu_short(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/linear/linear_short.yml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)

    def test_train_general_block_without_support(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/deform/general_block_wo_support.yml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 2.)

    def test_train_general_block(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/deform/general_block.yml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 2.)

    def test_train_general_block_input_selection(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/deform/general_block_input_selection.yml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 2.)

    @testing.attr.multi_gpu(2)
    def test_train_general_block_gpu(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/deform/general_block.yml')
        main_setting.trainer.gpu_id = 1
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 2.)

    def test_train_element_wise(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/linear/linear_element_wise.yml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)

    def test_train_element_batch(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/linear/linear_element_batch.yml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)

    def test_siml_updater_equivalent(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/linear/linear_element_batch.yml')

        main_setting.trainer.element_batch_size = 100000
        eb1_tr = trainer.Trainer(main_setting)
        if eb1_tr.setting.trainer.output_directory.exists():
            shutil.rmtree(eb1_tr.setting.trainer.output_directory)
        eb1_loss = eb1_tr.train()

        main_setting.trainer.element_batch_size = -1
        ebneg_tr = trainer.Trainer(main_setting)
        if ebneg_tr.setting.trainer.output_directory.exists():
            shutil.rmtree(ebneg_tr.setting.trainer.output_directory)
        ebneg_loss = ebneg_tr.train()

        main_setting.trainer.use_siml_updater = False
        std_tr = trainer.Trainer(main_setting)
        if std_tr.setting.trainer.output_directory.exists():
            shutil.rmtree(std_tr.setting.trainer.output_directory)
        std_loss = std_tr.train()

        np.testing.assert_almost_equal(eb1_loss, std_loss)
        np.testing.assert_almost_equal(ebneg_loss, std_loss)

    def test_train_element_learning_rate(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/linear/linear_short_lr.yml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)

    def test_infer_with_preprocessed_data(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/linear/pretrained/settings.yaml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        res = tr.infer(
            model_directory=Path('tests/data/linear/pretrained'),
            preprocessed_data_directory=Path(
                'tests/data/linear/preprocessed/validation'),
            converter_parameters_pkl=Path(
                'tests/data/linear/preprocessed/preprocessors.pkl'))
        np.testing.assert_almost_equal(
            res[0][1]['y'][0],
            np.load('tests/data/linear/interim/validation/0/y.npy'), decimal=3)
        np.testing.assert_array_less(res[0][2], 1e-7)

    def test_infer_with_raw_data(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/deform/pretrained/settings.yaml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)

        def conversion_function(fem_data, raw_directory=None):
            adj, _ = fem_data.calculate_adjacency_matrix_element()
            nadj = prepost.normalize_adjacency_matrix(adj)
            return {'adj': adj, 'nadj': nadj}

        res_from_raw = tr.infer(
            model_directory=Path('tests/data/deform/pretrained'),
            raw_data_directory=Path(
                'tests/data/deform/raw/test/tet2_4_modulusx0.9500'),
            converter_parameters_pkl=Path(
                'tests/data/deform/preprocessed/preprocessors.pkl'),
            conversion_function=conversion_function, save=False)
        res_from_preprocessed = tr.infer(
            model_directory=Path('tests/data/deform/pretrained'),
            preprocessed_data_directory=Path(
                'tests/data/deform/preprocessed/test/'
                'tet2_4_modulusx0.9500'),
            converter_parameters_pkl=Path(
                'tests/data/deform/preprocessed/preprocessors.pkl'))
        np.testing.assert_almost_equal(
            res_from_raw[0][1]['elemental_stress'][0],
            res_from_preprocessed[0][1]['elemental_stress'][0], decimal=3)
        np.testing.assert_almost_equal(
            res_from_raw[0][2], res_from_preprocessed[0][2])
        np.testing.assert_array_less(res_from_raw[0][2], 1e-2)

    def test_infer_with_raw_data_wo_answer(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/deform/pretrained/settings.yaml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)

        def conversion_function(fem_data, raw_directory=None):
            adj, _ = fem_data.calculate_adjacency_matrix_element()
            nadj = prepost.normalize_adjacency_matrix(adj)
            return {'adj': adj, 'nadj': nadj}

        res_from_raw = tr.infer(
            model_directory=Path('tests/data/deform/pretrained'),
            raw_data_directory=Path(
                'tests/data/deform/external/tet2_4_modulusx0.9500'),
            converter_parameters_pkl=Path(
                'tests/data/deform/preprocessed/preprocessors.pkl'),
            conversion_function=conversion_function, save=False)
        res_from_preprocessed = tr.infer(
            model_directory=Path('tests/data/deform/pretrained'),
            preprocessed_data_directory=Path(
                'tests/data/deform/preprocessed/test/'
                'tet2_4_modulusx0.9500'),
            converter_parameters_pkl=Path(
                'tests/data/deform/preprocessed/preprocessors.pkl'))
        np.testing.assert_almost_equal(
            res_from_raw[0][1]['elemental_stress'][0],
            res_from_preprocessed[0][1]['elemental_stress'][0], decimal=3)

    def test_infer_with_raw_data_wo_answer_with_model_file(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/deform/incomplete_pretrained/settings.yaml')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)

        def conversion_function(fem_data, raw_directory=None):
            adj, _ = fem_data.calculate_adjacency_matrix_element()
            nadj = prepost.normalize_adjacency_matrix(adj)
            return {'adj': adj, 'nadj': nadj}

        res_from_raw = tr.infer(
            model_file=Path(
                'tests/data/deform/incomplete_pretrained/snapshot_epoch_5000'),
            raw_data_directory=Path(
                'tests/data/deform/external/tet2_4_modulusx0.9500'),
            converter_parameters_pkl=Path(
                'tests/data/deform/preprocessed/preprocessors.pkl'),
            conversion_function=conversion_function, save=False)
        res_from_preprocessed = tr.infer(
            model_directory=Path('tests/data/deform/pretrained'),
            preprocessed_data_directory=Path(
                'tests/data/deform/preprocessed/test/'
                'tet2_4_modulusx0.9500'),
            converter_parameters_pkl=Path(
                'tests/data/deform/preprocessed/preprocessors.pkl'))
        np.testing.assert_almost_equal(
            res_from_raw[0][1]['elemental_stress'][0],
            res_from_preprocessed[0][1]['elemental_stress'][0], decimal=3)

    def test_infer_to_write_simulation_file(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            'tests/data/deform/incomplete_pretrained/settings.yaml')
        output_directory = Path('tests/data/deform/write_simulation')

        tr = trainer.Trainer(main_setting)
        if output_directory.exists():
            shutil.rmtree(output_directory)

        res_from_preprocessed = tr.infer(
            model_directory=Path('tests/data/deform/pretrained'),
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
            res_from_preprocessed[0][1]['elemental_stress'][0], decimal=7)

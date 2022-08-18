from pathlib import Path
import shutil
import unittest

import femio
import numpy as np
import torch

import siml.inferer as inferer
import siml.setting as setting
import siml.trainer as trainer


if torch.cuda.is_available():
    # GPU_ID = 0
    GPU_ID = -1
else:
    GPU_ID = -1


class TestGroups(unittest.TestCase):

    def test_group_simple(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation_thermal_stress/group_simple.yml'))
        main_setting.trainer.gpu_id = GPU_ID
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_group_repeat(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation_thermal_stress/group_repeat.yml'))
        main_setting.trainer.gpu_id = GPU_ID
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_heat_group_repeat(self):
        # NL solver repeat
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/heat_time_series/heat_group_nl_repeat.yml'))
        main_setting.trainer.gpu_id = GPU_ID
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss_implicit = tr.train()
        np.testing.assert_array_less(loss_implicit, 1.)

        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=main_setting.data.preprocessed_root)

        # Simple repeat
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/heat_time_series/heat_group_repeat.yml'))
        main_setting.trainer.gpu_id = GPU_ID
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss_repeat = tr.train()
        np.testing.assert_array_less(loss_repeat, 1.)

        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=main_setting.data.preprocessed_root)

        # No repeat
        main_setting_wo_repeat = setting.MainSetting.read_settings_yaml(
            Path('tests/data/heat_time_series/heat.yml'))
        main_setting_wo_repeat.trainer.gpu_id = GPU_ID
        tr_wo_repeat = trainer.Trainer(main_setting_wo_repeat)
        if tr_wo_repeat.setting.trainer.output_directory.exists():
            shutil.rmtree(tr_wo_repeat.setting.trainer.output_directory)
        loss_wo_repeat = tr_wo_repeat.train()
        np.testing.assert_array_less(loss_wo_repeat, .1)

        ir_wo_repeat = inferer.Inferer(
            main_setting_wo_repeat,
            converter_parameters_pkl=main_setting_wo_repeat
            .data.preprocessed_root
            / 'preprocessors.pkl')
        ir_wo_repeat.infer(
            model=main_setting_wo_repeat.trainer.output_directory,
            output_directory_base=tr_wo_repeat.setting
            .trainer.output_directory,
            data_directories=main_setting_wo_repeat.data.preprocessed_root)

        self.assertLess(loss_implicit, loss_repeat + 2.e-2)
        self.assertLess(loss_implicit, loss_wo_repeat + 3.e-2)

    def test_heat_boundary_repeat(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/heat_boundary/repeat.yml'))
        main_setting.trainer.gpu_id = GPU_ID
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss_repeat = tr.train()
        np.testing.assert_array_less(loss_repeat, 5.e-2)

        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=main_setting.data.preprocessed_root)

    def test_heat_boundary_implicit(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/heat_boundary/boundary_isogcn.yml'))
        main_setting.trainer.gpu_id = GPU_ID
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss_implicit = tr.train()
        np.testing.assert_array_less(loss_implicit, 5.e-2)

        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=main_setting.data.preprocessed_root)

    def test_heat_timeseries_better(self):
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/heat_time_series/heat_group_time_series.yml'))
        main_setting.trainer.gpu_id = GPU_ID
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, .1)

        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=Path(
                'tests/data/heat_time_series/preprocessed/2'))
        mse = np.mean((
            results[0]['dict_y']['t_10']
            - results[0]['dict_answer']['t_10'])**2)

        ref_main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/heat_time_series/heat_group_nl_repeat4.yml'))
        ref_main_setting.trainer.gpu_id = GPU_ID
        ref_tr = trainer.Trainer(ref_main_setting)
        if ref_tr.setting.trainer.output_directory.exists():
            shutil.rmtree(ref_tr.setting.trainer.output_directory)
        ref_tr.train()

        ref_ir = inferer.Inferer(
            ref_main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        ref_results = ref_ir.infer(
            model=ref_main_setting.trainer.output_directory,
            output_directory_base=ref_tr.setting.trainer.output_directory,
            data_directories=Path(
                'tests/data/heat_time_series/preprocessed/2'))
        ref_mse = np.mean((
            ref_results[0]['dict_y']['t_10']
            - ref_results[0]['dict_answer']['t_10'])**2)

        self.assertLess(mse, ref_mse + 5.e-2)

    def test_heat_timeseries_1step(self):
        main_setting_1step = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/heat_time_series/heat_group_time_series_1step.yml'))
        main_setting_1step.trainer.gpu_id = GPU_ID
        main_setting_1step.trainer.n_epoch = 10
        main_setting_1step.trainer.stop_trigger_epoch = 10
        tr_1step = trainer.Trainer(main_setting_1step)
        if tr_1step.setting.trainer.output_directory.exists():
            shutil.rmtree(tr_1step.setting.trainer.output_directory)
        loss_1step = tr_1step.train()

        ref_main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/heat_time_series/heat_group_nl_repeat.yml'))
        ref_main_setting.trainer.gpu_id = GPU_ID
        ref_main_setting.trainer.n_epoch = 10
        ref_main_setting.trainer.stop_trigger_epoch = 10
        ref_tr = trainer.Trainer(ref_main_setting)
        if ref_tr.setting.trainer.output_directory.exists():
            shutil.rmtree(ref_tr.setting.trainer.output_directory)
        ref_loss = ref_tr.train()

        np.testing.assert_almost_equal(loss_1step, ref_loss)

    def test_heat_timeseries_input_basic(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/heat_boundary/ts_isogcn.yml'))
        main_setting.trainer.gpu_id = GPU_ID
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        self.assertLess(loss, .1)

        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=main_setting.data.preprocessed_root)

    def test_heat_timeseries_input_slice(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/heat_boundary/ts_isogcn_slice.yml'))
        main_setting.trainer.gpu_id = GPU_ID
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        self.assertLess(loss, .1)

        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=main_setting.data.preprocessed_root)
        self.assertEqual(len(results[0]['dict_y']['ts_temperature']), 3)

    def test_heat_timeseries_input_split(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/heat_boundary/ts_isogcn_split.yml'))
        main_setting.trainer.gpu_id = GPU_ID
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        self.assertLess(loss, .1)

        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=main_setting.data.preprocessed_root)
        self.assertLess(results[0]['loss'], .5)

        fem_data = femio.read_files(
            'vtu', results[0]['output_directory'] / 'mesh.vtu')
        np.testing.assert_almost_equal(
            fem_data.nodal_data.get_attribute_data('input_ts_temperature_1'),
            fem_data.nodal_data.get_attribute_data('answer_ts_temperature_0'),
        )

    def test_heat_multigrid(self):
        ref_main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/heat_boundary/nl_rep10.yml'))
        ref_main_setting.trainer.gpu_id = GPU_ID
        ref_tr = trainer.Trainer(ref_main_setting)
        if ref_tr.setting.trainer.output_directory.exists():
            shutil.rmtree(ref_tr.setting.trainer.output_directory)
        ref_loss = ref_tr.train()
        np.testing.assert_array_less(ref_loss, .1)

        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/heat_boundary/multigrid.yml'))
        main_setting.trainer.gpu_id = GPU_ID
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, .1)
        self.assertLess(loss, ref_loss)

        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=Path(
                'tests/data/heat_boundary/preprocessed/cylinder/clscale0.3/'
                'steepness1.0_rep0'))

from pathlib import Path
import shutil
import unittest

from Cryptodome import Random
import numpy as np
import pandas as pd
import torch
import yaml

import siml.inferer as inferer
import siml.prepost as prepost
import siml.setting as setting
import siml.trainer as trainer


torch.autograd.set_detect_anomaly(True)


def conversion_function(fem_data, raw_directory=None):
    # To be used in test_preprocess_deform
    adj = fem_data.calculate_adjacency_matrix_element()
    nadj = prepost.normalize_adjacency_matrix(adj)
    x_grad, y_grad, z_grad = \
        fem_data.calculate_spatial_gradient_adjacency_matrices(
            'elemental')
    global_modulus = np.mean(
        fem_data.elemental_data.get_attribute_data('modulus'), keepdims=True)
    return {
        'adj': adj, 'nadj': nadj, 'global_modulus': global_modulus,
        'x_grad': x_grad, 'y_grad': y_grad, 'z_grad': z_grad}


class TestTrainer(unittest.TestCase):

    def test_train_cpu_short_on_memory(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_short.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)

    def test_train_cpu_short_lazy(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_short.yml'))
        main_setting.trainer.lazy = True
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)

    def test_train_general_block_without_support(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/general_block_wo_support.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 3.)

    def test_train_general_block(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/general_block.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 3.)

    def test_train_general_block_input_selection(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/general_block_input_selection.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 50.)

        # Confirm input feature dimension is as expected
        self.assertEqual(
            tr.model.dict_block['ResGCN1'].subchains[0][0].in_features, 6)
        self.assertEqual(tr.model.dict_block['MLP'].linears[0].in_features, 1)

    def test_train_element_wise(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_element_wise.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)
        self.assertEqual(len(tr.train_loader.dataset), 1000)
        self.assertEqual(tr.trainer.state.iteration, 1000 // 10 * 100)

    def test_train_element_batch(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_element_batch.yml'))
        tr_element_batch = trainer.Trainer(main_setting)
        if tr_element_batch.setting.trainer.output_directory.exists():
            shutil.rmtree(tr_element_batch.setting.trainer.output_directory)
        loss_element_batch = tr_element_batch.train()

        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_element_batch.yml'))
        main_setting.trainer.element_batch_size = -1
        main_setting.trainer.batch_size = 2
        tr_std = trainer.Trainer(main_setting)
        if tr_std.setting.trainer.output_directory.exists():
            shutil.rmtree(tr_std.setting.trainer.output_directory)
        loss_std = tr_std.train()

        self.assertLess(loss_element_batch, loss_std)

    def test_updater_equivalent(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_element_batch.yml'))

        main_setting.trainer.batch_size = 1
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

        np.testing.assert_almost_equal(eb1_loss, ebneg_loss)

    def test_train_element_learning_rate(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_short_lr.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)

    def test_gradient_consistency_with_padding(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_timeseries.yml'))
        main_setting.trainer.output_directory.mkdir(
            parents=True, exist_ok=True)
        tr = trainer.Trainer(main_setting)
        tr.prepare_training()
        x = np.reshape(np.arange(10*5*3), (10, 5, 3)).astype(np.float32) * .1
        y = torch.from_numpy((x[:, :, :2] * 2 - .5))

        tr.model.eval()
        pred_y_wo_padding = tr.model({'x': torch.from_numpy(x)})
        tr.optimizer.zero_grad()
        loss_wo_padding = tr.loss(
            pred_y_wo_padding, y, original_shapes=np.array([[10, 5]]))
        loss_wo_padding.backward(retain_graph=True)
        w_grad_wo_padding = tr.model.dict_block['Block'].linears[0].weight.grad
        b_grad_wo_padding = tr.model.dict_block['Block'].linears[0].bias.grad

        tr.optimizer.zero_grad()
        padded_x = np.concatenate([x, np.zeros((3, 5, 3))], axis=0).astype(
            np.float32)
        padded_y = np.concatenate([y, np.zeros((3, 5, 2))], axis=0).astype(
            np.float32)
        pred_y_w_padding = tr.model({'x': torch.from_numpy(padded_x)})
        loss_w_padding = tr.loss(
            pred_y_w_padding, torch.from_numpy(padded_y),
            original_shapes=np.array([[10, 5]]))
        loss_wo_padding.backward()
        w_grad_w_padding = tr.model.dict_block['Block'].linears[0].weight.grad
        b_grad_w_padding = tr.model.dict_block['Block'].linears[0].bias.grad

        np.testing.assert_almost_equal(
            loss_wo_padding.detach().numpy(), loss_w_padding.detach().numpy())
        np.testing.assert_almost_equal(
            w_grad_wo_padding.numpy(), w_grad_w_padding.numpy())
        np.testing.assert_almost_equal(
            b_grad_wo_padding.numpy(), b_grad_w_padding.numpy())

    def test_train_simplified_model(self):
        setting_yaml = Path('tests/data/simplified/mlp.yml')
        main_setting = setting.MainSetting.read_settings_yaml(setting_yaml)

        if main_setting.data.preprocessed_root.exists():
            shutil.rmtree(main_setting.data.preprocessed_root)
        preprocessor = prepost.Preprocessor.read_settings(setting_yaml)
        preprocessor.preprocess_interim_data()

        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 0.01)

    def test_evaluation_loss_not_depending_on_batch_size(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/mlp.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        main_setting.trainer.validation_batch_size = 1
        tr_batch_1 = trainer.Trainer(main_setting)
        loss_batch_1 = tr_batch_1.train()

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        main_setting.trainer.validation_batch_size = 2
        tr_batch_2 = trainer.Trainer(main_setting)
        loss_batch_2 = tr_batch_2.train()

        self.assertEqual(tr_batch_1.validation_loader.batch_size, 1)
        self.assertEqual(tr_batch_2.validation_loader.batch_size, 2)

        np.testing.assert_array_almost_equal(
            loss_batch_1, loss_batch_2, decimal=5)

    def test_early_stopping(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/use_pretrained.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        tr.train()
        self.assertLess(tr.trainer.state.epoch, main_setting.trainer.n_epoch)
        self.assertEqual(
            tr.trainer.state.epoch % main_setting.trainer.stop_trigger_epoch,
            0)

    def test_whole_processs(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/whole.yml'))
        shutil.rmtree(main_setting.data.interim_root, ignore_errors=True)
        shutil.rmtree(main_setting.data.preprocessed_root, ignore_errors=True)

        raw_converter = prepost.RawConverter(
            main_setting, conversion_function=conversion_function)
        raw_converter.convert()
        p = prepost.Preprocessor(main_setting)
        p.preprocess_interim_data()

        shutil.rmtree(
            main_setting.trainer.output_directory, ignore_errors=True)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1e-1)

        ir = inferer.Inferer(
            main_setting,
            model=main_setting.trainer.output_directory,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl',
            conversion_function=conversion_function, save=False)
        results = ir.infer(
            data_directories=main_setting.data.raw_root
            / 'train/tet2_3_modulusx0.9000', perform_preprocess=True)
        self.assertLess(results[0]['loss'], 1e-1)

    def test_whole_wildcard_processs(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/whole_wildcard.yml'))
        shutil.rmtree(main_setting.data.interim_root, ignore_errors=True)
        shutil.rmtree(main_setting.data.preprocessed_root, ignore_errors=True)

        raw_converter = prepost.RawConverter(
            main_setting, conversion_function=conversion_function)
        raw_converter.convert()
        p = prepost.Preprocessor(main_setting)
        p.preprocess_interim_data()

        shutil.rmtree(
            main_setting.trainer.output_directory, ignore_errors=True)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1e-1)

        ir = inferer.Inferer(
            main_setting,
            model=main_setting.trainer.output_directory,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl',
            conversion_function=conversion_function, save=False)
        results = ir.infer(
            data_directories=main_setting.data.raw_root
            / 'train/tet2_3_modulusx0.9000', perform_preprocess=True)
        self.assertLess(results[0]['loss'], 1e-1)

    def test_output_stats(self):
        original_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/general_block.yml'))
        original_tr = trainer.Trainer(original_setting)
        if original_tr.setting.trainer.output_directory.exists():
            shutil.rmtree(original_tr.setting.trainer.output_directory)
        original_loss = original_tr.train()

        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/output_stats.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()

        # Loss should not change depending on output_stats
        np.testing.assert_almost_equal(loss, original_loss, decimal=3)

        stats_file = tr.setting.trainer.output_directory \
            / 'stats_epoch37_iteration74.yml'
        with open(stats_file, 'r') as f:
            dict_data = yaml.load(f, Loader=yaml.SafeLoader)
        self.assertEqual(dict_data['iteration'], 74)

    def test_trainer_train_test_split(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/split/mlp.yml'))
        shutil.rmtree(
            main_setting.trainer.output_directory, ignore_errors=True)
        tr = trainer.Trainer(main_setting)
        tr.train()

        trained_setting = setting.MainSetting.read_settings_yaml(
            tr.setting.trainer.output_directory / 'settings.yml')
        self.assertEqual(
            len(trained_setting.data.train), 5)
        self.assertEqual(
            len(trained_setting.data.validation), 3)
        self.assertEqual(
            len(trained_setting.data.test), 2)

    def test_trainer_train_test_split_test0(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/split/mlp_test0.yml'))
        shutil.rmtree(
            main_setting.trainer.output_directory, ignore_errors=True)
        tr = trainer.Trainer(main_setting)
        tr.train()

        trained_setting = setting.MainSetting.read_settings_yaml(
            tr.setting.trainer.output_directory / 'settings.yml')
        self.assertEqual(
            len(trained_setting.data.train), 9)
        self.assertEqual(
            len(trained_setting.data.validation), 1)
        self.assertEqual(
            len(trained_setting.data.test), 0)

    def test_trainer_train_test_split_sample1(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/split/mlp_sample1.yml'))
        shutil.rmtree(
            main_setting.trainer.output_directory, ignore_errors=True)
        tr = trainer.Trainer(main_setting)
        tr.train()
        train_state, _ = tr.evaluate()
        train_loss = train_state.metrics['loss']

        trained_setting = setting.MainSetting.read_settings_yaml(
            tr.setting.trainer.output_directory / 'settings.yml')
        self.assertEqual(
            len(trained_setting.data.train), 1)
        self.assertEqual(
            len(trained_setting.data.validation), 0)
        self.assertEqual(
            len(trained_setting.data.test), 0)

        data_directory = main_setting.data.develop[0]  # pylint: disable=E1136
        ir = inferer.Inferer(
            main_setting,
            model=main_setting.trainer.output_directory,
            converter_parameters_pkl=data_directory.parent
            / 'preprocessors.pkl', save=False)
        results = ir.infer(data_directories=data_directory)
        np.testing.assert_almost_equal(results[0]['loss'], train_loss)

    def test_trainer_train_dict_input(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/dict_input.yml'))
        shutil.rmtree(
            main_setting.trainer.output_directory, ignore_errors=True)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_trainer_train_dict_input_w_support(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/dict_input_w_support.yml'))
        shutil.rmtree(
            main_setting.trainer.output_directory, ignore_errors=True)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_restart(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_short.yml'))

        # Complete training for reference
        complete_tr = trainer.Trainer(main_setting)
        complete_tr.setting.trainer.output_directory = Path(
            'tests/data/linear/linear_short_completed')
        if complete_tr.setting.trainer.output_directory.exists():
            shutil.rmtree(complete_tr.setting.trainer.output_directory)
        complete_tr.train()

        # Incomplete training
        incomplete_tr = trainer.Trainer(main_setting)
        incomplete_tr.setting.trainer.n_epoch = 20
        incomplete_tr.setting.trainer.output_directory = Path(
            'tests/data/linear/linear_short_incomplete')
        if incomplete_tr.setting.trainer.output_directory.exists():
            shutil.rmtree(incomplete_tr.setting.trainer.output_directory)
        incomplete_tr.train()

        # Restart training
        main_setting.trainer.restart_directory \
            = incomplete_tr.setting.trainer.output_directory
        restart_tr = trainer.Trainer(main_setting)
        restart_tr.setting.trainer.n_epoch = 100
        restart_tr.setting.trainer.output_directory = Path(
            'tests/data/linear/linear_short_restart')
        if restart_tr.setting.trainer.output_directory.exists():
            shutil.rmtree(restart_tr.setting.trainer.output_directory)
        loss = restart_tr.train()

        df = pd.read_csv(
            'tests/data/linear/linear_short_completed/log.csv',
            header=0, index_col=None, skipinitialspace=True)
        np.testing.assert_almost_equal(
            loss, df['validation_loss'].values[-1], decimal=3)

        restart_df = pd.read_csv(
            restart_tr.setting.trainer.output_directory / 'log.csv',
            header=0, index_col=None, skipinitialspace=True)
        self.assertEqual(len(restart_df.values), 8)

    def test_pretrain(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_short.yml'))
        tr_wo_pretrain = trainer.Trainer(main_setting)
        if tr_wo_pretrain.setting.trainer.output_directory.exists():
            shutil.rmtree(tr_wo_pretrain.setting.trainer.output_directory)
        loss_wo_pretrain = tr_wo_pretrain.train()

        main_setting.trainer.pretrain_directory \
            = tr_wo_pretrain.setting.trainer.output_directory
        tr_w_pretrain = trainer.Trainer(main_setting)
        if tr_w_pretrain.setting.trainer.output_directory.exists():
            shutil.rmtree(tr_w_pretrain.setting.trainer.output_directory)
        loss_w_pretrain = tr_w_pretrain.train()
        self.assertLess(loss_w_pretrain, loss_wo_pretrain)

    def test_whole_process_encryption(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/whole.yml'))
        main_setting.data.interim = [Path(
            'tests/data/deform/test_prepost/encrypt/interim')]
        main_setting.data.preprocessed = [Path(
            'tests/data/deform/test_prepost/encrypt/preprocessed')]

        main_setting.data.train = [Path(
            'tests/data/deform/test_prepost/encrypt/preprocessed/train')]
        main_setting.data.validation = [Path(
            'tests/data/deform/test_prepost/encrypt/preprocessed/validation')]
        main_setting.data.test = [Path(
            'tests/data/deform/test_prepost/encrypt/preprocessed/test')]

        main_setting.data.encrypt_key = Random.get_random_bytes(16)

        shutil.rmtree(main_setting.data.interim_root, ignore_errors=True)
        shutil.rmtree(main_setting.data.preprocessed_root, ignore_errors=True)

        raw_converter = prepost.RawConverter(
            main_setting, conversion_function=conversion_function)
        raw_converter.convert()
        p = prepost.Preprocessor(main_setting)
        p.preprocess_interim_data()

        with self.assertRaises(ValueError):
            np.load(
                main_setting.data.interim[0]
                / 'train/tet2_3_modulusx0.9000/elemental_strain.npy.enc')
        with self.assertRaises(OSError):
            np.load(
                main_setting.data.interim[0]
                / 'train/tet2_3_modulusx0.9000/elemental_strain.npy.enc',
                allow_pickle=True)

        with self.assertRaises(ValueError):
            np.load(
                main_setting.data.preprocessed[0]
                / 'train/tet2_3_modulusx0.9000/elemental_strain.npy.enc')
        with self.assertRaises(OSError):
            np.load(
                main_setting.data.preprocessed[0]
                / 'train/tet2_3_modulusx0.9000/elemental_strain.npy.enc',
                allow_pickle=True)

        shutil.rmtree(
            main_setting.trainer.output_directory, ignore_errors=True)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()

        ir = inferer.Inferer(
            main_setting,
            model=main_setting.trainer.output_directory,
            converter_parameters_pkl=main_setting.data.preprocessed[0]
            / 'preprocessors.pkl', save=False)
        results = ir.infer(
            data_directories=main_setting.data.preprocessed[0]
            / 'train/tet2_3_modulusx0.9000')
        self.assertLess(results[0]['loss'], loss * 5)

    def test_trainer_skip_dict_output(self):
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress/iso_gcn_skip_dict_output.yml'))
        shutil.rmtree(
            main_setting.trainer.output_directory, ignore_errors=True)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

        ir = inferer.Inferer(
            main_setting,
            model=main_setting.trainer.output_directory,
            converter_parameters_pkl=main_setting.data.preprocessed[0]
            / 'preprocessors.pkl', save=False)
        ir.setting.inferer.perform_inverse = False
        results = ir.infer(
            data_directories=Path(
                'tests/data/rotation_thermal_stress/preprocessed/'
                'cube/original'))

        t_mse_w_skip = np.mean((
            results[0]['dict_y']['cnt_temperature']
            - results[0]['dict_x']['cnt_temperature'])**2)

        # Traiing without skip
        main_setting.trainer.outputs['out_rank0'][0].skip = False
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

        ir = inferer.Inferer(
            main_setting,
            model=main_setting.trainer.output_directory,
            converter_parameters_pkl=main_setting.data.preprocessed[0]
            / 'preprocessors.pkl', save=False)
        ir.setting.inferer.perform_inverse = False
        results = ir.infer(
            data_directories=Path(
                'tests/data/rotation_thermal_stress/preprocessed/'
                'cube/original'))
        t_mse_wo_skip = np.mean((
            results[0]['dict_y']['cnt_temperature']
            - results[0]['dict_x']['cnt_temperature'])**2)

        print(t_mse_wo_skip, t_mse_w_skip)
        self.assertLess(t_mse_wo_skip, t_mse_w_skip)

    def test_trainer_skip_list_output(self):
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress/gcn_skip_list_output.yml'))
        shutil.rmtree(
            main_setting.trainer.output_directory, ignore_errors=True)
        tr = trainer.Trainer(main_setting)
        tr.train()

        ir = inferer.Inferer(
            main_setting,
            model=main_setting.trainer.output_directory,
            converter_parameters_pkl=main_setting.data.preprocessed[0]
            / 'preprocessors.pkl', save=False)
        ir.setting.inferer.perform_inverse = False
        results = ir.infer(
            data_directories=Path(
                'tests/data/rotation_thermal_stress/preprocessed/'
                'cube/original'))

        t_mse_w_skip = np.mean((
            results[0]['dict_y']['cnt_temperature']
            - results[0]['dict_x']['cnt_temperature'])**2)

        # Traiing without skip
        main_setting.trainer.outputs[0].skip = False
        tr = trainer.Trainer(main_setting)
        tr.train()

        ir = inferer.Inferer(
            main_setting,
            model=main_setting.trainer.output_directory,
            converter_parameters_pkl=main_setting.data.preprocessed[0]
            / 'preprocessors.pkl', save=False)
        ir.setting.inferer.perform_inverse = False
        results = ir.infer(
            data_directories=Path(
                'tests/data/rotation_thermal_stress/preprocessed/'
                'cube/original'))
        t_mse_wo_skip = np.mean((
            results[0]['dict_y']['cnt_temperature']
            - results[0]['dict_x']['cnt_temperature'])**2)

        print(t_mse_wo_skip, t_mse_w_skip)
        self.assertLess(t_mse_wo_skip, t_mse_w_skip)

    def test_residual_loss(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/heat_boundary/residual_loss.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss_implicit = tr.train()
        np.testing.assert_array_less(loss_implicit, 5.e-2)

from pathlib import Path
import shutil
import unittest

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
        self.assertEqual(len(tr.train_loader.dataset), 400)
        self.assertEqual(tr.trainer.state.iteration, 400 // 10 * 100)

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

        np.testing.assert_array_almost_equal(loss_batch_1, loss_batch_2)

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

        ir = inferer.Inferer(main_setting)
        results = ir.infer(
            model=Path('tests/data/deform/pretrained'),
            raw_data_directory=main_setting.data.raw_root
            / 'train/tet2_3_modulusx0.9000',
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl',
            conversion_function=conversion_function, save=False)
        self.assertLess(results[0]['loss'], loss)

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
            / 'stats_epoch37_iteration110.yml'
        with open(stats_file, 'r') as f:
            dict_data = yaml.load(f, Loader=yaml.SafeLoader)
        np.testing.assert_almost_equal(
            dict_data['dict_block.ResGCN2.subchains.0.1.bias']['grad_absmax'],
            0.8, decimal=1)
        self.assertEqual(dict_data['iteration'], 110)

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
        main_setting.trainer.restart_directory = Path(
            'tests/data/linear/linear_short_restart')
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        df = pd.read_csv(
            'tests/data/linear/linear_short_completed/log.csv',
            header=0, index_col=None, skipinitialspace=True)
        np.testing.assert_almost_equal(
            loss, df['validation_loss'].values[-1], decimal=5)

        restart_df = pd.read_csv(
            tr.setting.trainer.output_directory / 'log.csv',
            header=0, index_col=None, skipinitialspace=True)
        self.assertEqual(len(restart_df.values), 8)

    def test_pretrain(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_short.yml'))
        tr_wo_pretrain = trainer.Trainer(main_setting)
        if tr_wo_pretrain.setting.trainer.output_directory.exists():
            shutil.rmtree(tr_wo_pretrain.setting.trainer.output_directory)
        loss_wo_pretrain = tr_wo_pretrain.train()

        main_setting.trainer.pretrain_directory = Path(
            'tests/data/linear/linear_short_completed')
        tr_w_pretrain = trainer.Trainer(main_setting)
        if tr_w_pretrain.setting.trainer.output_directory.exists():
            shutil.rmtree(tr_w_pretrain.setting.trainer.output_directory)
        loss_w_pretrain = tr_w_pretrain.train()
        self.assertLess(loss_w_pretrain, loss_wo_pretrain)

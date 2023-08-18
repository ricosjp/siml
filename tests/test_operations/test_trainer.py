from unittest import mock
from pathlib import Path
import pickle
import shutil
import unittest
import pytest

from Cryptodome import Random
import numpy as np
import pandas as pd
import torch
import yaml

import siml.inferer as inferer
import siml.prepost as prepost
import siml.setting as setting
import siml.trainer as trainer
from siml.preprocessing import converter
from siml.preprocessing import ScalingConverter


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

    def test_train_cpu_short_debug_dataset(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_short.yml'))
        main_setting.trainer.debug_dataset = True
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)

    def test_train_cpu_short_lazy_shuffle_false(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_short.yml'))
        main_setting.trainer.lazy = True
        main_setting.trainer.train_data_shuffle = False
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)

    def test_train_cpu_short_output_loss_details(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_loss_details.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)

        output_csv = main_setting.trainer.output_directory / "log.csv"
        df = pd.read_csv(output_csv, skipinitialspace=True, header=0)

        train_check = (
            df.loc[:, "train_loss"] == df.loc[:, "train_loss_details/y"]
        ).to_numpy()
        assert np.all(train_check)

        validation_check = (
            df.loc[:, "validation_loss"]
            == df.loc[:, "validation_loss_details/y"]
        ).to_numpy()
        assert np.all(validation_check)

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
            tr._model.dict_block['ResGCN1'].subchains[0][0].in_features, 6)
        self.assertEqual(tr._model.dict_block['MLP'].linears[0].in_features, 1)

    def test_train_element_wise(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_element_wise.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)
        self.assertEqual(len(tr.train_loader.dataset), 1000)
        self.assertEqual(tr._trainer.state.iteration, 1000 // 10 * 100)

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
        x = np.reshape(np.arange(10*5*3), (10, 5, 3)).astype(np.float32) * .1
        y = torch.from_numpy((x[:, :, :2] * 2 - .5))

        tr._model.eval()
        pred_y_wo_padding = tr._model({'x': torch.from_numpy(x)})
        tr._optimizer.zero_grad()
        loss_wo_padding = tr._loss_calculator(
            pred_y_wo_padding, y, original_shapes=np.array([[10, 5]]))
        loss_wo_padding.backward(retain_graph=True)
        w_grad_wo_padding = tr._model.dict_block['Block'].linears[0].weight.grad  # NOQA
        b_grad_wo_padding = tr._model.dict_block['Block'].linears[0].bias.grad

        tr._optimizer.zero_grad()
        padded_x = np.concatenate([x, np.zeros((3, 5, 3))], axis=0).astype(
            np.float32)
        padded_y = np.concatenate([y, np.zeros((3, 5, 2))], axis=0).astype(
            np.float32)
        pred_y_w_padding = tr._model({'x': torch.from_numpy(padded_x)})
        loss_w_padding = tr._loss_calculator(
            pred_y_w_padding, torch.from_numpy(padded_y),
            original_shapes=np.array([[10, 5]]))
        loss_wo_padding.backward()
        w_grad_w_padding = tr._model.dict_block['Block'].linears[0].weight.grad
        b_grad_w_padding = tr._model.dict_block['Block'].linears[0].bias.grad

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
        preprocessor = ScalingConverter.read_settings(setting_yaml)
        preprocessor.fit_transform()

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
        self.assertLess(tr._trainer.state.epoch, main_setting.trainer.n_epoch)
        self.assertEqual(
            tr._trainer.state.epoch % main_setting.trainer.stop_trigger_epoch,
            0)

    def test_whole_processs(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/whole.yml'))
        shutil.rmtree(main_setting.data.interim_root, ignore_errors=True)
        shutil.rmtree(main_setting.data.preprocessed_root, ignore_errors=True)

        raw_converter = converter.RawConverter(
            main_setting, conversion_function=conversion_function)
        raw_converter.convert()
        p = ScalingConverter(main_setting)
        p.fit_transform()

        shutil.rmtree(
            main_setting.trainer.output_directory, ignore_errors=True)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1e-1)

        ir = inferer.WholeInferProcessor(
            main_setting,
            model_path=main_setting.trainer.output_directory,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl',
            conversion_function=conversion_function)
        results = ir.run(
            data_directories=main_setting.data.raw_root
            / 'train/tet2_3_modulusx0.9000',
            perform_preprocess=True,
            save_summary=False)
        self.assertLess(results[0]['loss'], 1e-1)

    def test_whole_wildcard_processs(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/whole_wildcard.yml'))
        shutil.rmtree(main_setting.data.interim_root, ignore_errors=True)
        shutil.rmtree(main_setting.data.preprocessed_root, ignore_errors=True)

        raw_converter = converter.RawConverter(
            main_setting, conversion_function=conversion_function)
        raw_converter.convert()
        p = ScalingConverter(main_setting)
        p.fit_transform()

        shutil.rmtree(
            main_setting.trainer.output_directory, ignore_errors=True)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1e-1)

        ir = inferer.WholeInferProcessor(
            main_setting,
            model_path=main_setting.trainer.output_directory,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl',
            conversion_function=conversion_function)
        results = ir.run(
            data_directories=main_setting.data.raw_root
            / 'train/tet2_3_modulusx0.9000',
            perform_preprocess=True,
            save_summary=False
        )
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
        train_state, _, _ = tr.evaluate()
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
        main_setting.inferer.save = False
        ir = inferer.Inferer(
            main_setting,
            model_path=main_setting.trainer.output_directory,
            converter_parameters_pkl=data_directory.parent
            / 'preprocessors.pkl')
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
        main_setting.trainer.output_directory = Path(
            'tests/data/linear/linear_short_completed')
        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        complete_tr = trainer.Trainer(main_setting)
        complete_tr.train()

        # Mock Incomplete training
        main_setting.trainer.n_epoch = 20
        main_setting.trainer.output_directory = Path(
            'tests/data/linear/linear_short_incomplete')
        incomplete_tr = trainer.Trainer(main_setting)
        if incomplete_tr.setting.trainer.output_directory.exists():
            shutil.rmtree(incomplete_tr.setting.trainer.output_directory)
        incomplete_tr.train()

        # Restart training
        main_setting.trainer.restart_directory \
            = incomplete_tr.setting.trainer.output_directory
        main_setting.trainer.output_directory = Path(
            'tests/data/linear/linear_short_restart')
        with mock.patch.object(
            setting.TrainerSetting,
            "n_epoch",
            new_callable=mock.PropertyMock,
            return_value=100
        ):
            restart_tr = trainer.Trainer(main_setting)
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

    def test_restart_overwrite(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_short.yml'))

        # Complete training for reference
        main_setting.trainer.output_directory = Path(
            'tests/data/linear/linear_short_completed')
        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)

        complete_tr = trainer.Trainer(main_setting)
        complete_tr.train()

        # Incomplete training
        main_setting.trainer.n_epoch = 20
        main_setting.trainer.output_directory = Path(
            'tests/data/linear/linear_short_incomplete')
        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        incomplete_tr = trainer.Trainer(main_setting)
        incomplete_tr.train()

        # Restart training
        main_setting.trainer.restart_directory \
            = incomplete_tr.setting.trainer.output_directory
        main_setting.trainer.output_directory \
            = incomplete_tr.setting.trainer.output_directory

        with mock.patch.object(
            setting.TrainerSetting,
            "n_epoch",
            new_callable=mock.PropertyMock,
            return_value=100
        ):
            restart_tr = trainer.Trainer(main_setting)
            loss = restart_tr.train()

        df = pd.read_csv(
            'tests/data/linear/linear_short_completed/log.csv',
            header=0, index_col=None, skipinitialspace=True)
        np.testing.assert_almost_equal(
            loss, df['validation_loss'].values[-1], decimal=3)

        print(restart_tr.setting.trainer.output_directory)
        restart_df = pd.read_csv(
            restart_tr.setting.trainer.output_directory / 'log.csv',
            header=0, index_col=None, skipinitialspace=True)
        self.assertEqual(len(restart_df.values), 10)

    def test_restart_multiple_times(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_short.yml'))
        # Incomplete training
        main_setting.trainer.n_epoch = 20
        main_setting.trainer.output_directory = Path(
            'tests/data/linear/linear_short_incomplete')
        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)

        incomplete_tr = trainer.Trainer(main_setting)
        incomplete_tr.train()

        # Restart training several times
        for n_epoch in [100, 120]:
            main_setting.trainer.restart_directory \
                = incomplete_tr.setting.trainer.output_directory
            main_setting.trainer.output_directory \
                = incomplete_tr.setting.trainer.output_directory
            with mock.patch.object(
                setting.TrainerSetting,
                "n_epoch",
                new_callable=mock.PropertyMock,
                return_value=n_epoch
            ):
                restart_tr = trainer.Trainer(main_setting)
                restart_tr.setting.trainer.n_epoch = n_epoch
                _ = restart_tr.train()

        print(restart_tr.setting.trainer.output_directory)
        restart_df = pd.read_csv(
            restart_tr.setting.trainer.output_directory / 'log.csv',
            header=0, index_col=None, skipinitialspace=True)
        self.assertEqual(len(restart_df.values), 12)

    def test_restart_yaml_files(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_short.yml'))
        # Mock Incomplete training
        main_setting.trainer.n_epoch = 20
        main_setting.trainer.output_directory = Path(
            'tests/data/linear/linear_short_incomplete')
        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        incomplete_tr = trainer.Trainer(main_setting)
        incomplete_tr.train()

        # Restart training several times
        output_dir = Path("tests/data/linear/linear_short_restart")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        for n_epoch in [100, 120]:
            with mock.patch.object(
                setting.TrainerSetting,
                "n_epoch",
                new_callable=mock.PropertyMock,
                return_value=n_epoch
            ):
                main_setting.trainer.restart_directory \
                    = incomplete_tr.setting.trainer.output_directory
                main_setting.trainer.output_directory = output_dir
                restart_tr = trainer.Trainer(main_setting)
                _ = restart_tr.train()

        yaml_files = list(output_dir.glob("*.yml*"))
        assert len(yaml_files) > 0

        for file_path in yaml_files:
            with open(file_path) as fr:
                content = yaml.safe_load(fr)
            assert content['trainer']['pretrain_directory'] is None

            # can load in inferer
            _ = inferer.Inferer.read_settings_file(file_path)

    def test_restart_deny(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_short.yml'))
        # Incomplete training
        main_setting.trainer.n_epoch = 20
        main_setting.trainer.output_directory = Path(
            'tests/data/linear/linear_short_incomplete')
        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        incomplete_tr = trainer.Trainer(main_setting)
        incomplete_tr.train()

        # Restart training
        with pytest.raises(FileExistsError):
            main_setting.trainer.restart_directory \
                = incomplete_tr.setting.trainer.output_directory
            main_setting.trainer.output_directory \
                = incomplete_tr.setting.trainer.output_directory
            restart_tr = trainer.Trainer(main_setting)
            _ = restart_tr.train()

        print(main_setting.trainer.output_directory)
        restart_df = pd.read_csv(
            main_setting.trainer.output_directory / 'log.csv',
            header=0, index_col=None, skipinitialspace=True)
        self.assertEqual(len(restart_df.values), 2)

    def test_pretrain(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_short.yml'))
        tr_wo_pretrain = trainer.Trainer(main_setting)
        if tr_wo_pretrain.setting.trainer.output_directory.exists():
            shutil.rmtree(tr_wo_pretrain.setting.trainer.output_directory)
        loss_wo_pretrain = tr_wo_pretrain.train()

        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_short.yml'))
        main_setting.trainer.pretrain_directory \
            = tr_wo_pretrain.setting.trainer.output_directory
        main_setting.trainer.output_directory = Path(
            'tests/data/linear/pretrained_tmp')
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

        raw_converter = converter.RawConverter(
            main_setting, conversion_function=conversion_function)
        raw_converter.convert()
        p = ScalingConverter(main_setting)
        p.fit_transform()

        with self.assertRaises(ValueError):
            np.load(
                main_setting.data.interim[0]
                / 'train/tet2_3_modulusx0.9000/elemental_strain.npy.enc')
        with self.assertRaises((pickle.UnpicklingError, OSError)):
            np.load(
                main_setting.data.interim[0]
                / 'train/tet2_3_modulusx0.9000/elemental_strain.npy.enc',
                allow_pickle=True)

        with self.assertRaises(ValueError):
            np.load(
                main_setting.data.preprocessed[0]
                / 'train/tet2_3_modulusx0.9000/elemental_strain.npy.enc')
        with self.assertRaises((pickle.UnpicklingError, OSError)):
            np.load(
                main_setting.data.preprocessed[0]
                / 'train/tet2_3_modulusx0.9000/elemental_strain.npy.enc',
                allow_pickle=True)

        shutil.rmtree(
            main_setting.trainer.output_directory, ignore_errors=True)
        main_setting.trainer.model_key = main_setting.data.encrypt_key
        tr = trainer.Trainer(main_setting)
        loss = tr.train()

        with self.assertRaises((pickle.UnpicklingError, UnicodeDecodeError)):
            torch.load(
                main_setting.trainer.output_directory
                / 'snapshot_epoch_100.pth.enc')

        with self.assertRaises(ValueError):
            setting.MainSetting.read_settings_yaml(
                main_setting.trainer.output_directory / 'settings.yml.enc')

        main_setting.inferer.save = False
        ir = inferer.Inferer.from_model_directory(
            main_setting.trainer.output_directory,
            decrypt_key=main_setting.trainer.model_key,
            converter_parameters_pkl=main_setting.data.preprocessed[0]
            / 'preprocessors.pkl'
        )
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

        main_setting.inferer.perform_inverse = False
        main_setting.inferer.save = False
        ir = inferer.Inferer(
            main_setting,
            model_path=main_setting.trainer.output_directory,
            converter_parameters_pkl=main_setting.data.preprocessed[0]
            / 'preprocessors.pkl'
        )
        results = ir.infer(
            data_directories=Path(
                'tests/data/rotation_thermal_stress/preprocessed/'
                'cube/original'))

        t_mse_w_skip = np.mean((
            results[0]['dict_y']['cnt_temperature']
            - results[0]['dict_answer']['cnt_temperature'])**2)

        # Traiing without skip
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress/iso_gcn_skip_dict_output.yml'))
        shutil.rmtree(
            main_setting.trainer.output_directory, ignore_errors=True)
        main_setting.trainer.outputs['out_rank0'][0].skip = False
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

        main_setting.inferer.perform_inverse = False
        ir = inferer.Inferer(
            main_setting,
            model_path=main_setting.trainer.output_directory,
            converter_parameters_pkl=main_setting.data.preprocessed[0]
            / 'preprocessors.pkl'
        )

        results = ir.infer(
            data_directories=Path(
                'tests/data/rotation_thermal_stress/preprocessed/'
                'cube/original'),
            save_summary=False
        )
        t_mse_wo_skip = np.mean((
            results[0]['dict_y']['cnt_temperature']
            - results[0]['dict_answer']['cnt_temperature'])**2)

        print(t_mse_wo_skip, t_mse_w_skip)
        self.assertLess(t_mse_wo_skip, t_mse_w_skip)

    def test_trainer_skip_list_output(self):
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress/gcn_skip_list_output.yml'))
        shutil.rmtree(
            main_setting.trainer.output_directory, ignore_errors=True)
        tr = trainer.Trainer(main_setting)
        tr.train()

        main_setting.inferer.perform_inverse = False
        ir = inferer.Inferer(
            main_setting,
            model_path=main_setting.trainer.output_directory,
            converter_parameters_pkl=main_setting.data.preprocessed[0]
            / 'preprocessors.pkl'
        )

        results = ir.infer(
            data_directories=Path(
                'tests/data/rotation_thermal_stress/preprocessed/'
                'cube/original'),
            save_summary=False
        )

        t_mse_w_skip = np.mean((
            results[0]['dict_y']['cnt_temperature']
            - results[0]['dict_answer']['cnt_temperature'])**2)

        # Traiing without skip
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress/gcn_skip_list_output.yml'))
        shutil.rmtree(
            main_setting.trainer.output_directory, ignore_errors=True)
        main_setting.trainer.outputs[0].skip = False
        tr = trainer.Trainer(main_setting)
        tr.train()

        main_setting.inferer.perform_inverse = False
        main_setting.inferer.save = False
        ir = inferer.Inferer(
            main_setting,
            model_path=main_setting.trainer.output_directory,
            converter_parameters_pkl=main_setting.data.preprocessed[0]
            / 'preprocessors.pkl'
        )
        results = ir.infer(
            data_directories=Path(
                'tests/data/rotation_thermal_stress/preprocessed/'
                'cube/original'))
        t_mse_wo_skip = np.mean((
            results[0]['dict_y']['cnt_temperature']
            - results[0]['dict_answer']['cnt_temperature'])**2)

        print(t_mse_wo_skip, t_mse_w_skip)
        self.assertLess(t_mse_wo_skip, t_mse_w_skip)

    def test_residual_loss(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/heat_boundary/residual_loss.yml'))
        tr = trainer.Trainer(main_setting)
        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        loss_implicit = tr.train()

        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/heat_boundary/residual_loss_coeff0.yml'))
        tr0 = trainer.Trainer(main_setting)
        if tr0.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss0 = tr0.train()

        ref_main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/heat_boundary/boundary.yml'))
        ref_tr = trainer.Trainer(ref_main_setting)
        if ref_tr.setting.trainer.output_directory.exists():
            shutil.rmtree(ref_tr.setting.trainer.output_directory)
        ref_loss_implicit = ref_tr.train()

        np.testing.assert_raises(
            AssertionError, np.testing.assert_almost_equal,
            loss_implicit, loss0, decimal=5)  # Assert almost not equal
        self.assertLess(
            tr._evaluator.state.metrics['GROUP1/residual'],
            tr0._evaluator.state.metrics['GROUP1/residual'])
        np.testing.assert_almost_equal(
            loss0, ref_loss_implicit, decimal=5)

    def test_trainer_train_with_user_loss_function(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/dict_input_user_loss.yml'))
        shutil.rmtree(
            main_setting.trainer.output_directory, ignore_errors=True)
        tr = trainer.Trainer(
            main_setting,
            user_loss_function_dic={
                "user_mspe": lambda x, y:
                torch.mean(torch.square((x - y) / (torch.norm(y) + 0.0001)))
            })
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_train_pseudo_batch(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/linear_short.yml'))
        main_setting.trainer.pseudo_batch_size = 5
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)

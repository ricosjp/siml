from pathlib import Path
import shutil
import unittest

import numpy as np
import torch

import siml.prepost as prepost
import siml.setting as setting
import siml.trainer as trainer


class TestTrainer(unittest.TestCase):

    def test_train_cpu_short(self):
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
            Path('tests/data/linear/linear.yml'))
        tr = trainer.Trainer(main_setting)
        tr.prepare_training()
        x = np.reshape(np.arange(5*3), (1, 5, 3)).astype(np.float32) * .1
        y = torch.from_numpy((x[:, :, :2] * 2 - .5))

        pred_y_wo_padding = tr.model({'x': torch.from_numpy(x)})
        tr.optimizer.zero_grad()
        loss_wo_padding = tr.loss(
            pred_y_wo_padding, y, original_shapes=[[5]])
        loss_wo_padding.backward(retain_graph=True)
        w_grad_wo_padding = tr.model.chains[0].linears[0].weight.grad
        b_grad_wo_padding = tr.model.chains[0].linears[0].bias.grad

        tr.optimizer.zero_grad()
        padded_x = np.concatenate([x, np.zeros((1, 2, 3))], axis=1).astype(
            np.float32)
        pred_y_w_padding = tr.model({'x': torch.from_numpy(padded_x)})
        loss_w_padding = tr.loss(
            pred_y_w_padding, y, original_shapes=[[5]])
        loss_wo_padding.backward()
        w_grad_w_padding = tr.model.chains[0].linears[0].weight.grad
        b_grad_w_padding = tr.model.chains[0].linears[0].bias.grad

        np.testing.assert_almost_equal(
            loss_wo_padding.detach().numpy(), loss_w_padding.detach().numpy())
        np.testing.assert_almost_equal(
            w_grad_wo_padding.numpy(), w_grad_w_padding.numpy())
        np.testing.assert_almost_equal(
            b_grad_wo_padding.numpy(), b_grad_w_padding.numpy())

    def test_train_simplified_model(self):
        setting_yaml = Path('tests/data/simplified/mlp.yml')
        main_setting = setting.MainSetting.read_settings_yaml(setting_yaml)

        if main_setting.data.preprocessed.exists():
            shutil.rmtree(main_setting.data.preprocessed)
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

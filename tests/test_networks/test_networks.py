from pathlib import Path
import shutil
import unittest

import numpy as np
import scipy.sparse as sp
from scipy.stats import ortho_group
import torch

import siml.inferer as inferer
import siml.networks.array2diagmat as array2diagmat
import siml.networks.array2symmat as array2symmat
import siml.networks.reducer as reducer
import siml.networks.reshape as reshape
import siml.networks.symmat2array as symmat2array
import siml.networks.tensor_operations as tensor_operations
import siml.networks.translator as translator
import siml.setting as setting
import siml.trainer as trainer


class TestNetworks(unittest.TestCase):

    def test_deepsets(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/deepsets.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)

    def test_deepsets_permutation(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/deepsets.yml'))
        tr = trainer.Trainer(main_setting)
        x = np.reshape(np.arange(5*3), (1, 5, 3)).astype(np.float32) * .1
        original_shapes = [[1, 5]]

        y_wo_permutation = tr._model({
            'x': torch.from_numpy(x), 'original_shapes': original_shapes})

        x_w_permutation = np.concatenate(
            [x[0, None, 2:], x[0, None, :2]], axis=1)
        y_w_permutation = tr._model({
            'x': torch.from_numpy(x_w_permutation),
            'original_shapes': original_shapes})

        np.testing.assert_almost_equal(
            np.concatenate(
                [
                    y_wo_permutation[0, None, 2:].detach().numpy(),
                    y_wo_permutation[0, None, :2].detach().numpy()],
                axis=1),
            y_w_permutation.detach().numpy())

    def test_res_gcn(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/res_gcn.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_gcn(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/gcn.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_incidence_gcn_heat(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/heat_time_series/incidence.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl',
            model_path=main_setting.trainer.output_directory
        )
        ir.infer(
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=main_setting.data.preprocessed_root)

    def test_incidence_gcn_deform(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/incidence.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_message_passing(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/message_passing.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_message_passing_non_concat(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/message_passing.yml'))
        main_setting.model.blocks[0].optional['concat'] = False
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_reducer_feature_direction(self):
        reducer_layer = reducer.Reducer(
            setting.BlockSetting(optional={'operator': 'mul'}))
        rank1 = np.random.rand(10, 3, 5)  # (n_node, dim, n_feature)
        rank0 = np.random.rand(10, 5)

        result = reducer_layer(
            torch.from_numpy(rank1), torch.from_numpy(rank0))
        desired = np.stack(
            [rank1[:, i_dim] * rank0 for i_dim in range(3)], axis=1)
        np.testing.assert_almost_equal(result, desired)

        swapped_result = reducer_layer(
            torch.from_numpy(rank0), torch.from_numpy(rank1))
        np.testing.assert_almost_equal(swapped_result, desired)

    def test_reducer_element_direction(self):
        reducer_layer = reducer.Reducer(
            setting.BlockSetting(optional={'operator': 'mul'}))
        n_batch = 2
        n_vertex_0 = 7
        n_vertex_1 = 13
        local_value_0 = np.random.rand(n_vertex_0, 5)
        local_value_1 = np.random.rand(n_vertex_1, 5)
        local_value = np.concatenate([local_value_0, local_value_1], axis=0)
        global_value = np.random.rand(n_batch, 5)
        original_shapes = [[n_vertex_0], [n_vertex_1]]
        result = reducer_layer(
            torch.from_numpy(local_value), torch.from_numpy(global_value),
            original_shapes=original_shapes)
        desired = np.concatenate([
            local_value_0 * global_value[0],
            local_value_1 * global_value[1]], axis=0)
        np.testing.assert_almost_equal(result, desired)

    def test_reduce(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/reduce.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_reduce_mlp(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/reduce_mlp.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_deform_gradient(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/res_gcn_grad.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_deform_gradient_share(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/res_gcn_grad.yml'))
        main_setting.model.blocks[0].optional['multiple_networks'] = False
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_train_time_series_simplified_data(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/simplified_timeseries/lstm.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, .1)

    def test_train_time_series_mesh_data_w_support(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform_timeseries/lstm_w_support.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1.)

    def test_train_time_series_mesh_data_wo_support(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform_timeseries/lstm_wo_support.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1.)

    def test_train_res_ltm(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform_timeseries/res_lstm.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1.)

    def test_train_tcn(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform_timeseries/tcn.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1.)

    def test_no_bias(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/no_bias.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)
        self.assertIsNone(tr._model.dict_block['Block'].linears[0].bias)

    def test_time_norm(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform_timeseries/time_norm.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1.)
        input_data = tr.train_loader.dataset[0]
        input_data = {'x': input_data['x']}
        out = tr._model(input_data)
        np.testing.assert_almost_equal(out.detach().numpy()[0], 0.)

    def test_integration_y1(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/ode/integration_y1_short.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1.)

        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl',
            model_path=main_setting.trainer.output_directory
        )
        results = ir.infer(
            data_directories=main_setting.data.preprocessed_root
            / 'test')
        self.assertLess(results[0]['loss'], 1.)

    def test_grad_grad_ridid_transformation_invariant(self):
        data_directory = Path(
            'tests/data/rotation/preprocessed/cube/clscale1.0/original')
        grad_x = sp.load_npz(data_directory / 'nodal_grad_x.npz')
        grad_y = sp.load_npz(data_directory / 'nodal_grad_y.npz')
        grad_z = sp.load_npz(data_directory / 'nodal_grad_z.npz')
        input_feature = np.reshape(
            np.arange(grad_x.shape[0] * 3), (grad_x.shape[0], 3)) * 5e-4
        grad_output_feature = \
            + grad_x.dot(grad_x.dot(input_feature)) \
            + grad_y.dot(grad_y.dot(input_feature)) \
            + grad_z.dot(grad_z.dot(input_feature))
        laplace = sp.load_npz(data_directory / 'nodal_laplacian.npz')
        laplace_output_feature = \
            laplace.dot(input_feature)
        # Due to numerical error, laplace_output_feature tends to have larger
        # error
        np.testing.assert_almost_equal(
            grad_output_feature - laplace_output_feature, 0., decimal=1)

        rotated_directory = Path(
            'tests/data/rotation/preprocessed/cube/clscale1.0/rotated')
        rotated_grad_x = sp.load_npz(rotated_directory / 'nodal_grad_x.npz')
        rotated_grad_y = sp.load_npz(rotated_directory / 'nodal_grad_y.npz')
        rotated_grad_z = sp.load_npz(rotated_directory / 'nodal_grad_z.npz')
        rotated_laplace = sp.load_npz(
            rotated_directory / 'nodal_laplacian.npz')
        rotated_grad_output_feature = \
            + rotated_grad_x.dot(rotated_grad_x.dot(input_feature)) \
            + rotated_grad_y.dot(rotated_grad_y.dot(input_feature)) \
            + rotated_grad_z.dot(rotated_grad_z.dot(input_feature))
        np.testing.assert_almost_equal(
            grad_output_feature, rotated_grad_output_feature, decimal=5)
        np.testing.assert_almost_equal(
            laplace.toarray(), rotated_laplace.toarray())

    def test_mlp_activation_after_residual(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/mlp_activation_after_residual.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)
        x = torch.from_numpy(np.random.rand(1, 4, 6).astype(np.float32))
        y = tr._model.dict_block['RES_MLP'](x)
        abs_residual = np.abs((y - x).detach().numpy())
        zero_fraction = np.sum(abs_residual < 1e-5) / abs_residual.size
        self.assertLess(.3, zero_fraction)

    def test_gcn_activation_after_residual(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/gcn_activation_after_residual.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)
        x = torch.from_numpy(np.random.rand(4, 6).astype(np.float32))
        _s = sp.coo_matrix([
            [1., 1., 0., 0.],
            [1., 1., 1., 1.],
            [0., 1., 1., 1.],
            [0., 1., 1., 1.],
        ], dtype=np.float32)
        s = [torch.sparse_coo_tensor(
            torch.stack([torch.from_numpy(_s.row), torch.from_numpy(_s.col)]),
            torch.from_numpy(_s.data), _s.shape)]
        y = tr._model.dict_block['RES_GCN'](x, supports=s)
        abs_residual = np.abs((y - x).detach().numpy())
        zero_fraction = np.sum(abs_residual < 1e-5) / abs_residual.size
        self.assertLess(.3, zero_fraction)

    def test_reduce_multiply(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/reduce_mul.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

        e = np.random.rand(1, 4, 1).astype(np.float32)
        epsilon = np.random.rand(1, 4, 6).astype(np.float32)
        sigma = tr._model.dict_block['MUL'](
            torch.from_numpy(e), torch.from_numpy(epsilon))
        np.testing.assert_almost_equal(
            sigma.detach().numpy(), e * epsilon)

    def test_gcn_with_array2symmat_symmat2array(self):
        _mat = np.load(
            'tests/data/rotation_thermal_stress/interim/cube/original'
            '/nodal_strain_mat.npy')
        mat = torch.from_numpy(np.concatenate([_mat, _mat * 2], axis=-1))
        _array = np.load(
            'tests/data/rotation_thermal_stress/interim/cube/original'
            '/nodal_strain_array.npy')
        array = torch.from_numpy(np.concatenate([_array, _array * 2], axis=-1))

        bs = setting.BlockSetting()
        bs.optional['to_engineering'] = True  # pylint: disable=E1137
        s2a = symmat2array.Symmat2Array(bs)
        a = s2a(mat)
        np.testing.assert_almost_equal(a.numpy(), array.numpy())

        a2s = array2symmat.Array2Symmat(setting.BlockSetting())
        s = a2s(array)
        np.testing.assert_almost_equal(s.numpy(), mat)

        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress'
            '/gcn_dict_input_dict_output.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

    def test_gcn_with_array2diagmat(self):
        array = np.random.rand(10, 4)

        bs = setting.BlockSetting()
        a2d = array2diagmat.Array2Diagmat(bs)
        diags = a2d(torch.from_numpy(array)).numpy()
        for i_vertex in range(array.shape[0]):
            for i_feature in range(array.shape[1]):
                np.testing.assert_almost_equal(
                    np.diag(diags[i_vertex, :, :, i_feature]),
                    array[i_vertex, i_feature])

    def test_contraction_2(self):
        contraction = tensor_operations.Contraction(setting.BlockSetting())
        a = np.random.rand(10, 3, 3, 3, 3, 5)
        b = np.random.rand(10, 3, 3, 5)
        desired = np.einsum('ijklmf,ilmf->ijkf', a, b)
        np.testing.assert_almost_equal(
            contraction(torch.from_numpy(a), torch.from_numpy(b)).numpy(),
            desired)
        np.testing.assert_almost_equal(
            contraction(torch.from_numpy(b), torch.from_numpy(a)).numpy(),
            desired)

    def test_contraction_1(self):
        contraction = tensor_operations.Contraction(
            setting.BlockSetting(activations=['sqrt']))
        a = np.random.rand(10, 3, 3, 3, 3, 5)
        desired = np.einsum('ijklmf,ijklmf->if', a, a)**.5
        np.testing.assert_almost_equal(
            contraction(torch.from_numpy(a)).numpy(),
            desired)

    def test_tensor_product(self):
        tensor_product = tensor_operations.TensorProduct(
            setting.BlockSetting())
        a = np.random.rand(10, 3, 3, 3, 3, 5)
        b = np.random.rand(10, 2, 2, 5)
        desired = np.einsum('ijklmf,inof->ijklmnof', a, b)
        np.testing.assert_almost_equal(
            tensor_product(torch.from_numpy(a), torch.from_numpy(b)).numpy(),
            desired)

    def test_translator_min_component_0_2(self):
        tor = translator.Translator(setting.BlockSetting(
            optional={'method': 'min', 'components': [0, 2]}))
        a = np.random.rand(10, 3)
        b = np.random.rand(20, 3)
        desired = np.concatenate([
            np.stack([
                a[:, 0] - np.min(a[:, 0]),
                a[:, 1],
                a[:, 2] - np.min(a[:, 2]),
            ], axis=-1),
            np.stack([
                b[:, 0] - np.min(b[:, 0]),
                b[:, 1],
                b[:, 2] - np.min(b[:, 2]),
            ], axis=-1),
        ])
        np.testing.assert_almost_equal(tor(
            torch.from_numpy(np.concatenate([a, b])),
            original_shapes=([10], [20])).numpy(), desired)

    def test_translator_min_component_all(self):
        tor = translator.Translator(setting.BlockSetting(
            optional={'method': 'mean'}))
        a = np.random.rand(10, 3)
        b = np.random.rand(20, 3)
        desired = np.concatenate([
            np.stack([
                a[:, 0] - np.mean(a[:, 0]),
                a[:, 1] - np.mean(a[:, 1]),
                a[:, 2] - np.mean(a[:, 2]),
            ], axis=-1),
            np.stack([
                b[:, 0] - np.mean(b[:, 0]),
                b[:, 1] - np.mean(b[:, 1]),
                b[:, 2] - np.mean(b[:, 2]),
            ], axis=-1),
        ])
        np.testing.assert_almost_equal(tor(
            torch.from_numpy(np.concatenate([a, b])),
            original_shapes=([10], [20])).numpy(), desired)

    def test_mlp_w_translate(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/mlp_w_translate.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 10.)

        x = np.random.rand(10 + 20, 6).astype(np.float32)
        t = tr._model.dict_block['TRANSLATE'](
            torch.from_numpy(x), original_shapes=[(10,), (20,)]
        ).detach().numpy()
        np.testing.assert_almost_equal(np.min(t[:, 0]), 0.)
        np.testing.assert_almost_equal(np.min(t[:, -1]), 0.)

    def test_pinv_mlp(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/pinv_mlp.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        tr.train()

        for l_ref, l_inv in zip(
                tr._model.dict_block['MLP'].linears,
                tr._model.dict_block['PINV_MLP'].linears[-1::-1]):
            np.testing.assert_almost_equal(
                l_inv.weight.detach().numpy(),
                np.linalg.pinv(l_ref.weight.detach().numpy()),
                decimal=5)
            np.testing.assert_almost_equal(
                l_inv.bias.detach().numpy(), - l_ref.bias.detach().numpy())

        x = torch.rand(100, 3, 3, 6)
        y = tr._model.dict_block['MLP'](x)
        x_ = tr._model.dict_block['PINV_MLP'](y)
        np.testing.assert_almost_equal(
            x_.detach().numpy(), x.detach().numpy(),
            decimal=5)

        x = torch.rand(100, 3, 3, 6) * 1e-2
        y = tr._model.dict_block['MLP'](x)
        x_ = tr._model.dict_block['PINV_MLP'](y)
        np.testing.assert_almost_equal(
            x_.detach().numpy() / 1e-2, x.detach().numpy() / 1e-2,
            decimal=3)

        x = torch.rand(100, 3, 3, 6) * 100
        y = tr._model.dict_block['MLP'](x)
        x_ = tr._model.dict_block['PINV_MLP'](y)
        np.testing.assert_almost_equal(
            x_.detach().numpy() / 100, x.detach().numpy() / 100,
            decimal=5)

    def test_pinv_mlp_no_bias(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/pinv_mlp_no_bias.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        tr.train()

        for l_ref, l_inv in zip(
                tr._model.dict_block['MLP'].linears,
                tr._model.dict_block['PINV_MLP'].linears[-1::-1]):
            np.testing.assert_almost_equal(
                l_inv.weight.detach().numpy(),
                np.linalg.pinv(l_ref.weight.detach().numpy()),
                decimal=5)
            self.assertEqual(l_inv.bias, 0)
            self.assertIsNone(l_ref.bias)

        x = torch.rand(100, 3, 3, 6)
        y = tr._model.dict_block['MLP'](x)
        x_ = tr._model.dict_block['PINV_MLP'](y)
        np.testing.assert_almost_equal(
            x_.detach().numpy(), x.detach().numpy(),
            decimal=5)

        x = torch.rand(100, 3, 3, 6) * 1e-2
        y = tr._model.dict_block['MLP'](x)
        x_ = tr._model.dict_block['PINV_MLP'](y)
        np.testing.assert_almost_equal(
            x_.detach().numpy() / 1e-2, x.detach().numpy() / 1e-2,
            decimal=3)

        x = torch.rand(100, 3, 3, 6) * 100
        y = tr._model.dict_block['MLP'](x)
        x_ = tr._model.dict_block['PINV_MLP'](y)
        np.testing.assert_almost_equal(
            x_.detach().numpy() / 100, x.detach().numpy() / 100,
            decimal=5)

    def test_share(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/share.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        tr.train()

        for l_ref, l_inv in zip(
                tr._model.dict_block['MLP'].linears,
                tr._model.dict_block['SHARE'].linears):
            np.testing.assert_almost_equal(
                l_inv.weight.detach().numpy(), l_ref.weight.detach().numpy())
            np.testing.assert_almost_equal(
                l_inv.bias.detach().numpy(), l_ref.bias.detach().numpy())

        x = torch.rand(100, 3, 3, 1)
        y = tr._model.dict_block['MLP'](x)
        y_ = tr._model.dict_block['SHARE'](x)
        np.testing.assert_almost_equal(
            y_.detach().numpy(), y.detach().numpy())

    def test_set_transformer_full(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/set_transformer.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

        # Test permutation invariance
        x = np.reshape(np.arange(5*12), (5, 12)).astype(np.float32) * .1
        original_shapes = [[5]]

        tr._model.eval()
        with torch.no_grad():
            y_wo_permutation = tr._model({
                'x': torch.from_numpy(x), 'original_shapes': original_shapes})

            x_w_permutation = np.concatenate(
                [x[2:], x[:2]], axis=0)
            y_w_permutation = tr._model({
                'x': torch.from_numpy(x_w_permutation),
                'original_shapes': original_shapes})

        np.testing.assert_almost_equal(
            y_wo_permutation.detach().numpy(),
            y_w_permutation.detach().numpy(), decimal=6)

    def test_set_transformer_encoder(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/set_transformer_encoder.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

        # Test permutation equivariance
        x = np.reshape(np.arange(5*7), (5, 7)).astype(np.float32) * .1
        original_shapes = [[5]]

        tr._model.eval()
        with torch.no_grad():
            y_wo_permutation = tr._model({
                'x': torch.from_numpy(x), 'original_shapes': original_shapes})

            x_w_permutation = np.concatenate(
                [x[2:], x[:2]], axis=0)
            y_w_permutation = tr._model({
                'x': torch.from_numpy(x_w_permutation),
                'original_shapes': original_shapes})

        np.testing.assert_almost_equal(
            np.concatenate(
                [
                    y_wo_permutation[2:].detach().numpy(),
                    y_wo_permutation[:2].detach().numpy()],
                axis=0),
            y_w_permutation.detach().numpy(), decimal=6)

    def test_equivariant_mlp_rank1(self):
        equivariant_mlp = tensor_operations.EquivariantMLP(
            setting.BlockSetting(
                nodes=[2, 4, 8], activations=['tanh', 'identity']))
        equivariant_mlp.eval()

        x = np.random.rand(200, 3, 2).astype(np.float32)
        with torch.no_grad():
            y = equivariant_mlp(torch.from_numpy(x)).detach().numpy()

        s = ortho_group.rvs(3).astype(np.float32)
        rotated_x = np.einsum('kl,ilf->ikf', s, x).astype(np.float32)
        with torch.no_grad():
            rotated_y = equivariant_mlp(
                torch.from_numpy(rotated_x)).detach().numpy()

        equivariant_y = np.einsum('kl,ilf->ikf', s, y)
        print(rotated_y[:5, :, 0])
        print(equivariant_y[:5, :, 0])
        np.testing.assert_almost_equal(
            np.mean((rotated_y - equivariant_y)**2)**.5
            / (np.mean(rotated_y**2)**.5 + 1e-5),
            0.)
        np.testing.assert_almost_equal(rotated_y, equivariant_y, decimal=6)

    def test_equivariant_mlp_rank2(self):
        equivariant_mlp = tensor_operations.EquivariantMLP(
            setting.BlockSetting(
                nodes=[2, 4, 8], activations=['tanh', 'tanh']))
        equivariant_mlp.eval()

        x = np.random.rand(200, 3, 3, 2).astype(np.float32)
        with torch.no_grad():
            y = equivariant_mlp(torch.from_numpy(x)).detach().numpy()

        s = ortho_group.rvs(3).astype(np.float32)
        rotated_x = np.einsum(
            'nm,ikmf->iknf', s, np.einsum('kl,ilmf->ikmf', s, x)
        ).astype(np.float32)
        with torch.no_grad():
            rotated_y = equivariant_mlp(
                torch.from_numpy(rotated_x)).detach().numpy()

        equivariant_y = np.einsum(
            'nm,ikmf->iknf', s, np.einsum('kl,ilmf->ikmf', s, y))
        print(rotated_y[:5, :, :, 0])
        print(equivariant_y[:5, :, :, 0])
        np.testing.assert_almost_equal(
            np.mean((rotated_y - equivariant_y)**2)**.5
            / (np.mean(rotated_y**2)**.5 + 1e-5), 0., decimal=6)
        np.testing.assert_almost_equal(rotated_y, equivariant_y, decimal=6)

    def test_reducer_batch_broadchast(self):
        t1 = torch.rand(15, 3, 8)
        t2 = torch.rand(3, 8)
        original_shapes = {
            't1': np.array([[10], [3], [2]]),
            't2': np.array([[1], [1], [1]]),
        }
        reducer_ = reducer.Reducer(
            setting.BlockSetting(
                type='reducer',
                optional={'operator': 'add', 'split_keys': ['t1', 't2']}))
        t = reducer_(
            t1, t2, op='add', original_shapes=original_shapes).detach().numpy()

        desired_t = torch.cat([
            t1[:10] + t2[[0]],
            t1[10:10 + 3] + t2[[1]],
            t1[10 + 3:] + t2[[2]],
        ], dim=0).numpy()
        np.testing.assert_array_almost_equal(t, desired_t)

    def test_features_to_time_series(self):
        r = np.random.rand(10, 3, 1)
        x = np.concatenate([
            r, 10 * r, 100 * r, 1000 * r, 10000 * r], axis=-1)
        desired_y = np.stack([
            r, 10 * r, 100 * r, 1000 * r, 10000 * r], axis=0)
        to_ts = reshape.FeaturesToTimeSeries(setting.BlockSetting())
        y = to_ts(torch.from_numpy(x)).detach().numpy()
        np.testing.assert_almost_equal(y, desired_y)

        to_f = reshape.TimeSeriesToFeatures(setting.BlockSetting(is_last=True))
        reversed_y = to_f(torch.from_numpy(y)).detach().numpy()
        np.testing.assert_almost_equal(reversed_y, x)

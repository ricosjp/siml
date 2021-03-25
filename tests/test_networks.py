from pathlib import Path
import shutil
import unittest

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import torch

import siml.inferer as inferer
import siml.networks.activations as activations
import siml.networks.array2diagmat as array2diagmat
import siml.networks.array2symmat as array2symmat
import siml.networks.reducer as reducer
import siml.networks.symmat2array as symmat2array
import siml.networks.tensor_operations as tensor_operations
import siml.networks.translator as translator
import siml.setting as setting
import siml.trainer as trainer


PLOT = False


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
        tr.prepare_training()
        x = np.reshape(np.arange(5*3), (1, 5, 3)).astype(np.float32) * .1
        original_shapes = [[1, 5]]

        y_wo_permutation = tr.model({
            'x': torch.from_numpy(x), 'original_shapes': original_shapes})

        x_w_permutation = np.concatenate(
            [x[0, None, 2:], x[0, None, :2]], axis=1)
        y_w_permutation = tr.model({
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

    def test_activations(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/activations.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 1.)

    def test_mish(self):
        np.testing.assert_almost_equal(
            activations.mish(torch.Tensor([100.])), [100.])
        np.testing.assert_almost_equal(
            activations.mish(torch.Tensor([-100.])), [0.])
        np.testing.assert_almost_equal(
            activations.mish(torch.Tensor([1.])),
            [1. * np.tanh(np.log(1 + np.exp(1.)))])
        if PLOT:
            x = np.linspace(-10., 10., 100)
            mish = activations.mish(torch.from_numpy(x))
            plt.plot(x, mish.numpy())
            plt.show()

    def test_no_bias(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/linear/no_bias.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)
        self.assertIsNone(tr.model.dict_block['Block'].linears[0].bias)

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
        out = tr.model(input_data)
        np.testing.assert_almost_equal(out.detach().numpy()[0], 0.)

    def test_integration_y1(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/ode/integration_y1_short.yml'))

        if main_setting.trainer.output_directory.exists():
            shutil.rmtree(main_setting.trainer.output_directory)
        tr = trainer.Trainer(main_setting)
        loss = tr.train()
        self.assertLess(loss, 5.e-1)

        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            data_directories=main_setting.data.preprocessed_root
            / 'test')
        self.assertLess(results[0]['loss'], 1.)

    def test_grad_gcn(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/grad_gcn.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)
        self.assertEqual(len(tr.model.dict_block['GRAD_GCN1'].subchains), 1)
        self.assertEqual(len(tr.model.dict_block['GRAD_GCN2'].subchains), 1)

    def test_grad_res_gcn(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/grad_res_gcn.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)
        self.assertEqual(len(tr.model.dict_block['GRAD_GCN1'].subchains), 1)
        self.assertEqual(len(tr.model.dict_block['GRAD_GCN2'].subchains), 1)

        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/deform/grad_res_gcn/settings.yml'))
        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=Path(
                'tests/data/deform/preprocessed/preprocessors.pkl'))
        ir.setting.inferer.output_directory = \
            main_setting.trainer.output_directory / 'inferred'
        if ir.setting.inferer.output_directory.exists():
            shutil.rmtree(ir.setting.inferer.output_directory)

        res = ir.infer(
            model=Path('tests/data/deform/grad_res_gcn'),
            data_directories=Path(
                'tests/data/deform/preprocessed/validation/'
                'tet2_3_modulusx0.9500'))
        np.testing.assert_array_less(res[0]['loss'], 1e-2)

    def test_laplace_net(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation/laplace_net.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)
        self.assertEqual(len(tr.model.dict_block['LAPLACE_NET1'].subchains), 1)
        self.assertEqual(len(tr.model.dict_block['LAPLACE_NET2'].subchains), 1)

    def test_res_laplace_net(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation/res_laplace_net.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)
        self.assertEqual(len(tr.model.dict_block['LAPLACE_NET1'].subchains), 1)
        self.assertEqual(len(tr.model.dict_block['LAPLACE_NET2'].subchains), 1)

        # Confirm results does not change under rigid body transformation
        grad_grad_ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=Path(
                'tests/data/rotation/preprocessed/preprocessors.pkl'))
        grad_grad_ir.setting.inferer.output_directory = \
            main_setting.trainer.output_directory / 'inferred'
        if grad_grad_ir.setting.inferer.output_directory.exists():
            shutil.rmtree(grad_grad_ir.setting.inferer.output_directory)
        results = grad_grad_ir.infer(
            model=main_setting.trainer.output_directory,
            data_directories=[
                Path(
                    'tests/data/rotation/preprocessed/cube/clscale1.0/'
                    'original'),
                Path(
                    'tests/data/rotation/preprocessed/cube/clscale1.0/'
                    'rotated')])
        np.testing.assert_almost_equal(
            results[0]['dict_y']['t_100'],
            results[1]['dict_y']['t_100'], decimal=5)

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
        y = tr.model.dict_block['RES_MLP'](x)
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
        y = tr.model.dict_block['RES_GCN'](x, s)
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
        sigma = tr.model.dict_block['MUL'](
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
        t = tr.model.dict_block['TRANSLATE'](
            torch.from_numpy(x), original_shapes=[(10,), (20,)]
        ).detach().numpy()
        np.testing.assert_almost_equal(np.min(t[:, 0]), 0.)
        np.testing.assert_almost_equal(np.min(t[:, -1]), 0.)

    def test_normalize(self):
        a = np.random.rand(10, 3)
        normalized_a = activations.normalize(torch.from_numpy(a)).numpy()
        np.testing.assert_almost_equal(
            normalized_a, a / (np.linalg.norm(a, axis=1)[..., None] + 1e-5))

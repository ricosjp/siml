
from pathlib import Path
import shutil
import sys
import unittest

import femio
import numpy as np
import scipy.sparse as sp
import torch

import siml.datasets as datasets
import siml.inferer as inferer
import siml.networks.boundary as boundary
import siml.networks.iso_gcn as iso_gcn
import siml.networks.mlp as mlp
import siml.setting as setting
import siml.trainer as trainer

sys.path.append('tests')
import preprocess  # NOQA


PLOT = False


class TestBoundary(unittest.TestCase):

    def test_neumann_encoder_linear(self):
        data_path = Path('tests/data/grad/interim/train/0')
        linear = mlp.MLP(
            setting.BlockSetting(
                type='mlp', nodes=[1, 1], bias=False))
        iso_gcn_ = iso_gcn.IsoGCN(
            setting.BlockSetting(
                type='iso_gcn',
                bias=False,
                support_input_indices=[0, 1, 2],
                optional={
                    'propagations': ['convolution'],
                    'create_subchain': False}))
        neumann_encoder = boundary.NeumannEncoder(
            setting.BlockSetting(
                type='neumann_encoder',
                input_names=['ID_RANK0', 'IN_NEUMANN']),
            reference_block=linear)
        neumann_iso_gcn = boundary.NeumannIsoGCN(
            setting.BlockSetting(
                type='neumann_isogcn',
                input_names=['ISOGCN', 'NEUMANN_ENCODER', 'IN_MOMENT']),
            reference_block=iso_gcn_)

        gx = sp.load_npz(data_path / 'nodal_grad_x.npz')
        gy = sp.load_npz(data_path / 'nodal_grad_y.npz')
        gz = sp.load_npz(data_path / 'nodal_grad_z.npz')
        supports = datasets.convert_sparse_tensor([
            datasets.pad_sparse(gx),
            datasets.pad_sparse(gy),
            datasets.pad_sparse(gz)])
        np_phi = np.load(data_path / 'phi.npy').astype(np.float32)
        phi = torch.from_numpy(np_phi)
        np_minv = np.load(data_path / 'inversed_moment_tensor.npy').astype(
            np.float32)
        minv = torch.from_numpy(np_minv)
        np_directed_neumann = np.load(
            data_path / 'directed_neumann.npy').astype(np.float32)
        directed_neumann = torch.from_numpy(np_directed_neumann)
        np_surface_normal = np.load(
            data_path / 'nodal_surface_normal.npy').astype(np.float32)
        boundary_filter = np.linalg.norm(
            np_surface_normal[..., 0], axis=1) > .5
        np.testing.assert_almost_equal(
            np.linalg.norm(np_surface_normal[boundary_filter, ..., 0], axis=1),
            1.)
        np_desired_grad = np.load(data_path / 'grad.npy')

        plain_desired_grad_wo_neumann = np.stack([
            gx.dot(np_phi), gy.dot(np_phi), gz.dot(np_phi)], axis=1)
        plain_desired_grad_w_neumann = plain_desired_grad_wo_neumann \
            + np.einsum('ijkf,ikf->ijf', np_minv, directed_neumann)
        plain_desired_grad_w_neumann = plain_desired_grad_wo_neumann \
            + np.einsum('ijkf,ikf->ijf', np_minv, directed_neumann)
        plain_actual_grad_wo_neumann = iso_gcn_(phi, supports=supports)
        plain_actual_grad_w_neumann = neumann_iso_gcn(
            plain_actual_grad_wo_neumann, directed_neumann,
            minv).detach().numpy()
        np.testing.assert_almost_equal(
            plain_actual_grad_wo_neumann.detach().numpy(),
            plain_desired_grad_wo_neumann)
        np.testing.assert_almost_equal(
            plain_actual_grad_w_neumann, plain_desired_grad_w_neumann)
        np.testing.assert_almost_equal(
            plain_actual_grad_w_neumann, np_desired_grad, decimal=1)
        fem_data = femio.read_directory(
            'ucd', Path('tests/data/grad/raw/train/0'))
        fem_data.nodal_data.update_data(
            fem_data.nodes.ids, {
                'true_grad': np_desired_grad,
                'error_pred_true_w_neumann':
                (plain_actual_grad_w_neumann - np_desired_grad)[
                    ..., 0],
                'error_pred_true_wo_neumann': (
                    plain_actual_grad_wo_neumann.detach().numpy()
                    - np_desired_grad)[..., 0],
                'desired_grad_wo_neumann':
                plain_desired_grad_wo_neumann[..., 0],
                'actual_grad_wo_neumann':
                plain_actual_grad_wo_neumann.detach().numpy()[..., 0],
                'desired_grad_w_neumann':
                    plain_desired_grad_w_neumann[..., 0],
                'actual_grad_w_neumann':
                plain_actual_grad_w_neumann[..., 0]})
        fem_data.write(
            'ucd',
            'tests/data/grad/output/test_neumann_encoder_linear/mesh.inp',
            overwrite=True)

        lineared_phi = linear(phi)
        encoded_neumann = neumann_encoder(
            phi, directed_neumann)
        grad_wo_neumann = iso_gcn_(lineared_phi, supports=supports)
        np_grad_wo_neumann = grad_wo_neumann.detach().numpy()
        np_grad_w_neumann = neumann_iso_gcn(
            grad_wo_neumann, encoded_neumann, minv).detach().numpy()

        desired_grad_wo_neumann = plain_desired_grad_wo_neumann \
            * linear.linears[0].weight.detach().numpy()[0, 0]
        desired_grad_w_neumann = plain_desired_grad_w_neumann \
            * linear.linears[0].weight.detach().numpy()[0, 0]
        np.testing.assert_almost_equal(
            np_grad_wo_neumann, desired_grad_wo_neumann, decimal=6)
        np.testing.assert_almost_equal(
            np_grad_w_neumann, desired_grad_w_neumann, decimal=6)

        error_wo_neumann = np.mean(np.linalg.norm(
            np_grad_wo_neumann - desired_grad_w_neumann, axis=1)[
                boundary_filter])
        error_w_neumann = np.mean(np.linalg.norm(
            np_grad_w_neumann - desired_grad_w_neumann, axis=1)[
                boundary_filter])
        self.assertLess(error_w_neumann, error_wo_neumann)

    def test_neumann_weighted_isogcn(self):
        data_path = Path('tests/data/grad/interim/train/0')
        iso_gcn_ = iso_gcn.IsoGCN(
            setting.BlockSetting(
                type='iso_gcn',
                nodes=[1, 8],
                bias=False,
                support_input_indices=[0, 1, 2],
                optional={
                    'propagations': ['convolution']}))
        neumann_iso_gcn = boundary.NeumannIsoGCN(
            setting.BlockSetting(
                type='neumann_isogcn',
                input_names=['ISOGCN', 'NEUMANN_ENCODER', 'IN_MOMENT']),
            reference_block=iso_gcn_)

        gx = sp.load_npz(data_path / 'nodal_grad_x.npz')
        gy = sp.load_npz(data_path / 'nodal_grad_y.npz')
        gz = sp.load_npz(data_path / 'nodal_grad_z.npz')
        supports = datasets.convert_sparse_tensor([
            datasets.pad_sparse(gx),
            datasets.pad_sparse(gy),
            datasets.pad_sparse(gz)])
        np_phi = np.load(data_path / 'phi.npy').astype(np.float32)
        phi = torch.from_numpy(np_phi)
        np_minv = np.load(data_path / 'inversed_moment_tensor.npy').astype(
            np.float32)
        minv = torch.from_numpy(np_minv)
        np_directed_neumann = np.load(
            data_path / 'directed_neumann.npy').astype(np.float32)
        directed_neumann = torch.from_numpy(np_directed_neumann)
        np_surface_normal = np.load(
            data_path / 'nodal_surface_normal.npy').astype(np.float32)
        boundary_filter = np.linalg.norm(
            np_surface_normal[..., 0], axis=1) > .5
        np.testing.assert_almost_equal(
            np.linalg.norm(np_surface_normal[boundary_filter, ..., 0], axis=1),
            1.)

        grad_wo_neumann = iso_gcn_(phi, supports=supports)
        np_grad_wo_neumann = grad_wo_neumann.detach().numpy()
        np_grad_w_neumann = neumann_iso_gcn(
            grad_wo_neumann, directed_neumann, minv).detach().numpy()

        plain_desired_grad_wo_neumann = np.stack([
            gx.dot(np_phi), gy.dot(np_phi), gz.dot(np_phi)], axis=1)
        plain_desired_grad_w_neumann = plain_desired_grad_wo_neumann \
            + np.einsum('ijkf,ikf->ijf', np_minv, directed_neumann)
        desired_grad_w_neumann = np.einsum(
            'ikf,fg->ikg',
            plain_desired_grad_w_neumann,
            iso_gcn_.subchains[0][0].weight.detach().numpy().T)
        np.testing.assert_almost_equal(
            np_grad_w_neumann, desired_grad_w_neumann, decimal=6)
        fem_data = femio.read_directory(
            'ucd', Path('tests/data/grad/raw/train/0'))
        fem_data.nodal_data.update_data(
            fem_data.nodes.ids, {
                'desired_grad_w_neumann':
                    desired_grad_w_neumann[..., 0],
                'actual_grad_w_neumann':
                np_grad_w_neumann[..., 0]})
        fem_data.write(
            'ucd',
            'tests/data/grad/output/test_neumann_weighted_isogcn/mesh.inp',
            overwrite=True)

        error_wo_neumann = np.mean(np.linalg.norm(
            np_grad_wo_neumann - desired_grad_w_neumann, axis=1)[
                boundary_filter])
        error_w_neumann = np.mean(np.linalg.norm(
            np_grad_w_neumann - desired_grad_w_neumann, axis=1)[
                boundary_filter])
        self.assertLess(error_w_neumann, error_wo_neumann)

    def test_neumann_encoder_nonlinear(self):
        data_path = Path('tests/data/grad/interim/train/0')
        mlp_ = mlp.MLP(
            setting.BlockSetting(
                type='mlp', nodes=[1, 16, 16, 1],
                activations=['tanh', 'tanh', 'identity']))
        iso_gcn_ = iso_gcn.IsoGCN(
            setting.BlockSetting(
                type='iso_gcn',
                bias=False,
                support_input_indices=[0, 1, 2],
                optional={
                    'propagations': ['convolution'],
                    'create_subchain': False}))
        neumann_encoder = boundary.NeumannEncoder(
            setting.BlockSetting(
                type='neumann_encoder',
                input_names=['ID_RANK0', 'IN_NEUMANN', 'NORMALS']),
            reference_block=mlp_)
        neumann_iso_gcn = boundary.NeumannIsoGCN(
            setting.BlockSetting(
                type='neumann_isogcn',
                input_names=['ISOGCN', 'NEUMANN_ENCODER', 'IN_MOMENT']),
            reference_block=iso_gcn_)

        gx = sp.load_npz(data_path / 'nodal_grad_x.npz')
        gy = sp.load_npz(data_path / 'nodal_grad_y.npz')
        gz = sp.load_npz(data_path / 'nodal_grad_z.npz')
        supports = datasets.convert_sparse_tensor([
            datasets.pad_sparse(gx),
            datasets.pad_sparse(gy),
            datasets.pad_sparse(gz)])
        np_phi = np.load(data_path / 'phi.npy').astype(np.float32)
        phi = torch.from_numpy(np_phi)
        np_minv = np.load(data_path / 'inversed_moment_tensor.npy').astype(
            np.float32)
        minv = torch.from_numpy(np_minv)
        np_directed_neumann = np.load(
            data_path / 'directed_neumann.npy').astype(np.float32)
        np_neumann = np.load(
            data_path / 'neumann.npy').astype(np.float32)
        neumann = torch.from_numpy(np_neumann)
        np_surface_normal = np.load(
            data_path / 'nodal_surface_normal.npy').astype(np.float32)
        np_weighted_normal = np.load(
            data_path / 'nodal_weighted_normal.npy').astype(np.float32)
        weighted_normal = torch.from_numpy(np_weighted_normal)
        boundary_filter = np.linalg.norm(
            np_directed_neumann[..., 0], axis=1) > 1.e-2

        encoded_phi = mlp_(phi)
        encoded_neumann = neumann_encoder(
            encoded_phi, neumann, weighted_normal)
        grad_wo_neumann = iso_gcn_(encoded_phi, supports=supports)
        np_grad_wo_neumann = grad_wo_neumann.detach().numpy()
        np_grad_w_neumann = neumann_iso_gcn(
            grad_wo_neumann, encoded_neumann, minv).detach().numpy()

        actual_neumann = np.einsum(
            'ijf,ij->if', np_grad_w_neumann, np_surface_normal[..., 0])
        neumann_wo_neumann = np.einsum(
            'ijf,ij->if', np_grad_wo_neumann, np_surface_normal[..., 0])
        fem_data = femio.read_directory(
            'ucd', Path('tests/data/grad/raw/train/0'))
        # np_desired_grad = fem_data.nodal_data.get_attribute_data('grad')[
        #     ..., None]
        renormalized_encoded_neumann = encoded_neumann.detach().numpy() \
            / np.linalg.norm(np_weighted_normal, axis=1)[..., None]
        desired_neumann = np.einsum(
            'ijf,ij->if', renormalized_encoded_neumann,
            np_surface_normal[..., 0])
        fem_data.nodal_data.update_data(
            fem_data.nodes.ids, {
                'renormalized_encoded_neumann':
                renormalized_encoded_neumann[..., 0],
                'actual_grad_wo_neumann':
                np_grad_wo_neumann[..., 0],
                'actual_grad_w_neumann':
                np_grad_w_neumann[..., 0]})
        fem_data.write(
            'ucd',
            'tests/data/grad/output/test_neumann_encoder_nonlinear/mesh.inp',
            overwrite=True)
        np.testing.assert_almost_equal(
            actual_neumann[boundary_filter],
            desired_neumann[boundary_filter], decimal=2)
        rela_error_neumann = np.mean((
            actual_neumann[boundary_filter]
            - desired_neumann[boundary_filter])**2)**.5 \
            / np.mean(desired_neumann[boundary_filter]**2)**.5
        rela_error_wo_neumann = np.mean((
            neumann_wo_neumann[boundary_filter]
            - desired_neumann[boundary_filter])**2)**.5 \
            / np.mean(desired_neumann[boundary_filter]**2)**.5

        self.assertLess(rela_error_neumann, rela_error_wo_neumann)

    def test_grad_neumann_identity(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/grad/identity.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, .5)

        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        ir.setting.inferer.write_simulation = True
        ir.setting.inferer.write_simulation_base = Path('tests/data/grad/raw')
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=main_setting.data.test[0])
        x = results[0]['dict_x']
        y = results[0]['dict_y']
        filter_boundary = np.linalg.norm(
            x['directed_neumann'][..., 0], axis=1) > 1.e-2
        normal = np.load(
            results[0]['data_directory'] / 'nodal_surface_normal.npy')
        answer_boundary = np.einsum(
            'ijf,ijf->i', x['grad'], normal)[filter_boundary]
        prediction_boundary = np.einsum(
            'ijf,ijf->i', y['grad'], normal)[filter_boundary]
        error_boundary = np.mean(
            (prediction_boundary - answer_boundary)**2)**.5 \
            / np.mean(answer_boundary**2)**.5
        np.testing.assert_array_almost_equal(
            prediction_boundary, answer_boundary, decimal=1)

        main_setting_wo_boundary = setting.MainSetting.read_settings_yaml(
            Path('tests/data/grad/identity_wo_boundary.yml'))
        tr_wo_boundary = trainer.Trainer(main_setting_wo_boundary)
        if tr_wo_boundary.setting.trainer.output_directory.exists():
            shutil.rmtree(tr_wo_boundary.setting.trainer.output_directory)
        loss = tr_wo_boundary.train()
        np.testing.assert_array_less(loss, 1.)

        ir_wo_boundary = inferer.Inferer(
            main_setting_wo_boundary,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        ir_wo_boundary.setting.inferer.write_simulation = True
        ir_wo_boundary.setting.inferer.write_simulation_base = Path(
            'tests/data/grad/raw')
        results_wo_boundary = ir_wo_boundary.infer(
            model=main_setting_wo_boundary.trainer.output_directory,
            output_directory_base=tr_wo_boundary.setting
            .trainer.output_directory,
            data_directories=main_setting_wo_boundary.data.test[0])
        y_wo_boundary = results_wo_boundary[0]['dict_y']
        prediction_wo_boundary = np.einsum(
            'ijf,ijf->i', y_wo_boundary['grad'], normal)[filter_boundary]

        error_wo_boundary = np.mean(
            (prediction_wo_boundary - answer_boundary)**2)**.5 \
            / np.mean(answer_boundary**2)**.5

        self.assertLess(error_boundary, error_wo_boundary)

    def test_grad_neumann_linear(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/grad/linear.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, .5)

        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        ir.setting.inferer.write_simulation = True
        ir.setting.inferer.write_simulation_base = Path('tests/data/grad/raw')
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=main_setting.data.test[0])
        x = results[0]['dict_x']
        y = results[0]['dict_y']
        filter_boundary = np.linalg.norm(
            x['directed_neumann'][..., 0], axis=1) > 1.e-2
        normal = np.load(
            results[0]['data_directory'] / 'nodal_surface_normal.npy')
        answer_boundary = np.einsum(
            'ijf,ijf->i', x['grad'], normal)[filter_boundary]
        prediction_boundary = np.einsum(
            'ijf,ijf->i', y['grad'], normal)[filter_boundary]
        error_boundary = np.mean(
            (prediction_boundary - answer_boundary)**2)**.5 \
            / np.mean(answer_boundary**2)**.5
        np.testing.assert_array_almost_equal(
            prediction_boundary, answer_boundary, decimal=1)

        main_setting_wo_boundary = setting.MainSetting.read_settings_yaml(
            Path('tests/data/grad/linear_wo_boundary.yml'))
        tr_wo_boundary = trainer.Trainer(main_setting_wo_boundary)
        if tr_wo_boundary.setting.trainer.output_directory.exists():
            shutil.rmtree(tr_wo_boundary.setting.trainer.output_directory)
        loss = tr_wo_boundary.train()
        np.testing.assert_array_less(loss, 1.)

        ir_wo_boundary = inferer.Inferer(
            main_setting_wo_boundary,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        ir_wo_boundary.setting.inferer.write_simulation = True
        ir_wo_boundary.setting.inferer.write_simulation_base = Path(
            'tests/data/grad/raw')
        results_wo_boundary = ir_wo_boundary.infer(
            model=main_setting_wo_boundary.trainer.output_directory,
            output_directory_base=tr_wo_boundary.setting
            .trainer.output_directory,
            data_directories=main_setting_wo_boundary.data.test[0])
        y_wo_boundary = results_wo_boundary[0]['dict_y']
        prediction_wo_boundary = np.einsum(
            'ijf,ijf->i', y_wo_boundary['grad'], normal)[filter_boundary]

        error_wo_boundary = np.mean(
            (prediction_wo_boundary - answer_boundary)**2)**.5 \
            / np.mean(answer_boundary**2)**.5

        self.assertLess(error_boundary, error_wo_boundary)

    def test_grad_neumann_identity_equivariance(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/grad/identity.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, .5)

        # Test equivariance
        ir = inferer.Inferer(
            main_setting,
            conversion_function=preprocess.conversion_function_grad,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=Path('tests/data/grad/raw/test/0'),
            perform_preprocess=True)
        y = results[0]['dict_y']['grad']

        rotated_test_directory = Path('tests/data/grad/raw/rotated_test/0')
        rotated_results = ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=rotated_test_directory,
            perform_preprocess=True)
        rotated_y = rotated_results[0]['dict_y']['grad']

        rotation = np.load(rotated_test_directory / 'rotation.npy')
        np.testing.assert_almost_equal(
            np.einsum('kl,ilf->ikf', rotation, y), rotated_y, decimal=6)

    def test_grad_neumann_nonlinear(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/grad/nonlinear.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, .05)

        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        ir.setting.inferer.write_simulation = True
        ir.setting.inferer.write_simulation_base = Path('tests/data/grad/raw')
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=main_setting.data.test[0])
        x = results[0]['dict_x']
        y = results[0]['dict_y']
        filter_boundary = np.linalg.norm(
            x['nodal_weighted_normal'][..., 0], axis=1) > 1e-5
        normal = np.load(
            results[0]['data_directory'] / 'nodal_surface_normal.npy')
        answer_boundary = np.einsum(
            'ijf,ijf->i', x['grad'], normal)[filter_boundary]
        prediction_boundary = np.einsum(
            'ijf,ijf->i', y['grad'], normal)[filter_boundary]
        error_boundary = np.mean(
            (prediction_boundary - answer_boundary)**2)**.5 \
            / np.mean(answer_boundary**2)**.5

        main_setting_wo_boundary = setting.MainSetting.read_settings_yaml(
            Path('tests/data/grad/nonlinear_wo_boundary.yml'))
        tr_wo_boundary = trainer.Trainer(main_setting_wo_boundary)
        if tr_wo_boundary.setting.trainer.output_directory.exists():
            shutil.rmtree(tr_wo_boundary.setting.trainer.output_directory)
        loss = tr_wo_boundary.train()
        np.testing.assert_array_less(loss, 1.)

        ir_wo_boundary = inferer.Inferer(
            main_setting_wo_boundary,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        ir_wo_boundary.setting.inferer.write_simulation = True
        ir_wo_boundary.setting.inferer.write_simulation_base = Path(
            'tests/data/grad/raw')
        results_wo_boundary = ir_wo_boundary.infer(
            model=main_setting_wo_boundary.trainer.output_directory,
            output_directory_base=tr_wo_boundary.setting
            .trainer.output_directory,
            data_directories=main_setting_wo_boundary.data.test[0])
        y_wo_boundary = results_wo_boundary[0]['dict_y']
        prediction_wo_boundary = np.einsum(
            'ijf,ijf->i', y_wo_boundary['grad'], normal)[filter_boundary]

        error_wo_boundary = np.mean(
            (prediction_wo_boundary - answer_boundary)**2)**.5 \
            / np.mean(answer_boundary**2)**.5

        self.assertLess(error_boundary, error_wo_boundary)

    def test_grad_neumann_nonlinear_merged(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/grad/nonlinear_merged.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, .05)

        ir = inferer.Inferer(
            main_setting,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=main_setting.data.test[0])
        y = results[0]['dict_y']

        ref_main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/grad/nonlinear.yml'))
        ref_tr = trainer.Trainer(ref_main_setting)
        if ref_tr.setting.trainer.output_directory.exists():
            shutil.rmtree(ref_tr.setting.trainer.output_directory)
        ref_loss = ref_tr.train()

        np.testing.assert_almost_equal(loss, ref_loss)

        ref_ir = inferer.Inferer(
            ref_main_setting,
            converter_parameters_pkl=ref_main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        ref_results = ref_ir.infer(
            model=ref_main_setting.trainer.output_directory,
            output_directory_base=ref_tr.setting.trainer.output_directory,
            data_directories=ref_main_setting.data.test[0])
        ref_y = ref_results[0]['dict_y']

        np.testing.assert_almost_equal(y['grad'], ref_y['grad'])

    def test_grad_neumann_nonlinear_coeff(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/grad/nonlinear_merged_coeff.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, .05)

        # Test equivariance
        ir = inferer.Inferer(
            main_setting,
            conversion_function=preprocess.conversion_function_grad,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=Path('tests/data/grad/raw/test/0'),
            perform_preprocess=True)
        y = results[0]['dict_y']['grad']

        rotated_test_directory = Path('tests/data/grad/raw/rotated_test/0')
        rotated_results = ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=rotated_test_directory,
            perform_preprocess=True)
        rotated_y = rotated_results[0]['dict_y']['grad']

        rotation = np.load(rotated_test_directory / 'rotation.npy')
        np.testing.assert_almost_equal(
            np.einsum('kl,ilf->ikf', rotation, y), rotated_y, decimal=2)

    def test_heat_interaction(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/heat_interaction/isogcn.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, .05)

        ir = inferer.Inferer(
            main_setting,
            conversion_function=preprocess
            .conversion_function_heat_interaction,
            converter_parameters_pkl=main_setting.data.preprocessed_root
            / 'preprocessors.pkl')
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            output_directory_base=tr.setting.trainer.output_directory,
            data_directories=Path(
                'tests/data/heat_interaction/preprocessed/9'),
            perform_preprocess=False)

        validation_path = Path('tests/data/heat_interaction/raw/9')
        fem_data_1 = femio.read_files('ucd', validation_path / 'mesh_1.inp')
        fem_data_1.nodal_data.update_data(
            fem_data_1.nodes.ids, {'pred': results[0]['dict_y']['phi_1_1']})
        fem_data_1.write('ucd', results[0]['output_directory'] / 'mesh_1.inp')

        fem_data_2 = femio.read_files('ucd', validation_path / 'mesh_2.inp')
        fem_data_2.nodal_data.update_data(
            fem_data_2.nodes.ids, {'pred': results[0]['dict_y']['phi_1_2']})
        fem_data_2.write('ucd', results[0]['output_directory'] / 'mesh_2.inp')
        np.testing.assert_array_less(results[0]['loss'], .05)

    def test_assignment_different_shape(self):
        x = torch.rand(10, 3)
        y = torch.rand(10, 3)
        np_cond = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 1])
        cond = torch.from_numpy(np_cond[..., None])
        assignment = boundary.Assignment(setting.BlockSetting())
        res = assignment(x, y, cond).detach().numpy()
        np.testing.assert_almost_equal(
            res[np_cond > .5], y.numpy()[np_cond > .5])
        np.testing.assert_almost_equal(
            res[np_cond <= .5], x.numpy()[np_cond <= .5])

    def test_assignment_same_shape(self):
        x = torch.rand(10, 3)
        y = torch.rand(10, 3)
        np_cond = np.random.rand(10, 3)
        cond = torch.from_numpy(np_cond)
        assignment = boundary.Assignment(setting.BlockSetting())
        res = assignment(x, y, cond).detach().numpy()
        np.testing.assert_almost_equal(
            res[np_cond > .5], y.numpy()[np_cond > .5])
        np.testing.assert_almost_equal(
            res[np_cond <= .5], x.numpy()[np_cond <= .5])

    def test_assignment_broadcast(self):
        x = torch.rand(10, 3)
        y = torch.rand(1, 1)
        np_cond = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 1])
        cond = torch.from_numpy(np_cond[..., None])
        assignment = boundary.Assignment(setting.BlockSetting(
            optional={'broadcast': True}))
        res = assignment(x, y, cond, original_shapes=[[10]]).detach().numpy()
        np.testing.assert_almost_equal(res[np_cond > .5], y.numpy()[0, 0])
        np.testing.assert_almost_equal(
            res[np_cond <= .5], x[np_cond <= .5].numpy())

    def test_assignment_broadcast_multi_batch(self):
        x = torch.rand(10, 3)
        y = torch.rand(3, 1)
        np_cond = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 1])
        cond = torch.from_numpy(np_cond[..., None])
        assignment = boundary.Assignment(setting.BlockSetting(
            optional={'broadcast': True}))
        res = assignment(
            x, y, cond, original_shapes=[[4], [3], [3]]).detach().numpy()
        np.testing.assert_almost_equal(
            res[:4][np_cond[:4] > .5], y.numpy()[0, 0])
        np.testing.assert_almost_equal(
            res[4:4+3][np_cond[4:4+3] > .5], y.numpy()[1, 0])
        np.testing.assert_almost_equal(
            res[7:][np_cond[7:] > .5], y.numpy()[2, 0])
        np.testing.assert_almost_equal(
            res[np_cond <= .5], x[np_cond <= .5].numpy())

    def test_assignment_broadcast_multi_batch_dict_input(self):
        x = torch.rand(10, 3)
        y = torch.rand(3, 1)
        np_cond = np.array([0, 1, 0, 0, 1, 1, 1, 0, 1, 1])
        cond = torch.from_numpy(np_cond[..., None])
        assignment = boundary.Assignment(setting.BlockSetting(
            optional={'broadcast': True, 'dict_key': 'x'}))
        res = assignment(
            x, y, cond, original_shapes={'x': [[4], [3], [3]]}
        ).detach().numpy()
        np.testing.assert_almost_equal(
            res[:4][np_cond[:4] > .5], y.numpy()[0, 0])
        np.testing.assert_almost_equal(
            res[4:4+3][np_cond[4:4+3] > .5], y.numpy()[1, 0])
        np.testing.assert_almost_equal(
            res[7:][np_cond[7:] > .5], y.numpy()[2, 0])
        np.testing.assert_almost_equal(
            res[np_cond <= .5], x[np_cond <= .5].numpy())

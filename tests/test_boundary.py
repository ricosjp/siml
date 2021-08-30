
from pathlib import Path
import shutil
import unittest

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
            np_directed_neumann[..., 0], axis=1) > 1.e-2

        plain_desired_neumann = np.einsum(
            'ijf,ijf->if', np_directed_neumann, np_surface_normal)
        plain_actual_directed_neumann = np.stack([
            gx.dot(np_phi), gy.dot(np_phi), gz.dot(np_phi)], axis=1) \
            + np.einsum('ijkf,ikf->ijf', np_minv, directed_neumann)
        plain_actual_neumann = np.einsum(
            'ijf,ijf->if', plain_actual_directed_neumann, np_surface_normal)
        np.testing.assert_almost_equal(
            plain_actual_neumann[boundary_filter],
            plain_desired_neumann[boundary_filter], decimal=2)
        rela_error_plain_neumann = np.mean((
            plain_actual_neumann[boundary_filter]
            - plain_desired_neumann[boundary_filter])**2)**.5 \
            / np.mean(plain_desired_neumann[boundary_filter]**2)**.5

        lineared_phi = linear(phi)
        encoded_neumann = neumann_encoder(
            phi, directed_neumann)
        grad_wo_neumann = iso_gcn_(lineared_phi, supports)
        np_grad_wo_neumann = grad_wo_neumann.detach().numpy()
        np_grad_w_neumann = neumann_iso_gcn(
            grad_wo_neumann, encoded_neumann, minv).detach().numpy()

        desired_neumann = np.einsum(
            'ijf,ijf->if', np_directed_neumann, np_surface_normal
        ) * linear.linears[0].weight.detach().numpy()[0, 0]
        actual_neumann = np.einsum(
            'ijf,ijf->if', np_grad_w_neumann, np_surface_normal)
        neumann_wo_neumann = np.einsum(
            'ijf,ijf->if', np_grad_wo_neumann, np_surface_normal)
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

        np.testing.assert_almost_equal(
            rela_error_neumann, rela_error_plain_neumann, decimal=5)
        self.assertLess(rela_error_neumann, rela_error_wo_neumann)

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
        directed_neumann = torch.from_numpy(np_directed_neumann)
        np_neumann = np.load(
            data_path / 'neumann.npy').astype(np.float32)
        neumann = torch.from_numpy(np_neumann)
        np_surface_normal = np.load(
            data_path / 'nodal_surface_normal.npy').astype(np.float32)
        surface_normal = torch.from_numpy(np_surface_normal)
        boundary_filter = np.linalg.norm(
            np_directed_neumann[..., 0], axis=1) > 1.e-2

        plain_desired_neumann = np.einsum(
            'ijf,ijf->if', np_directed_neumann, np_surface_normal)
        plain_actual_directed_neumann = np.stack([
            gx.dot(np_phi), gy.dot(np_phi), gz.dot(np_phi)], axis=1) \
            + np.einsum('ijkf,ikf->ijf', np_minv, directed_neumann)
        plain_actual_neumann = np.einsum(
            'ijf,ijf->if', plain_actual_directed_neumann, np_surface_normal)
        np.testing.assert_almost_equal(
            plain_actual_neumann[boundary_filter],
            plain_desired_neumann[boundary_filter], decimal=2)
        rela_error_plain_neumann = np.mean((
            plain_actual_neumann[boundary_filter]
            - plain_desired_neumann[boundary_filter])**2)**.5 \
            / np.mean(plain_desired_neumann[boundary_filter]**2)**.5

        encoded_phi = mlp_(phi)
        encoded_neumann = neumann_encoder(
            phi, neumann, surface_normal)
        grad_wo_neumann = iso_gcn_(encoded_phi, supports)
        np_grad_wo_neumann = grad_wo_neumann.detach().numpy()
        np_grad_w_neumann = neumann_iso_gcn(
            grad_wo_neumann, encoded_neumann, minv).detach().numpy()

        desired_neumann = np.einsum(
            'ijf,ij->if', encoded_neumann.detach().numpy(),
            np_surface_normal[..., 0])
        actual_neumann = np.einsum(
            'ijf,ij->if', np_grad_w_neumann, np_surface_normal[..., 0])
        neumann_wo_neumann = np.einsum(
            'ijf,ij->if', np_grad_wo_neumann, np_surface_normal[..., 0])
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

        np.testing.assert_almost_equal(
            rela_error_neumann, rela_error_plain_neumann, decimal=3)
        self.assertLess(rela_error_neumann, rela_error_wo_neumann)

    def test_grad_neumann_linear(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/grad/linear.yml'))
        # main_setting = setting.MainSetting.read_settings_yaml(
        #     Path('tests/data/grad/identity.yml'))
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
        # np.testing.assert_array_almost_equal(
        #     prediction_boundary, answer_boundary)

        main_setting_wo_boundary = setting.MainSetting.read_settings_yaml(
            Path('tests/data/grad/linear_wo_boundary.yml'))
        # main_setting_wo_boundary = setting.MainSetting.read_settings_yaml(
        #     Path('tests/data/grad/identity_wo_boundary.yml'))
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
            x['nodal_surface_normal'][..., 0], axis=1) > .5
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

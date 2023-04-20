
from pathlib import Path
import shutil

import numpy as np
import scipy.sparse as sp
import torch

import siml.datasets as datasets
import siml.inferer as inferer
import siml.networks.iso_gcn as iso_gcn
import siml.networks.penn as penn
import siml.setting as setting
import siml.trainer as trainer

import equivariance_base as equivariance_base
import preprocess

PLOT = False


class TestPENN(equivariance_base.EquivarianceBase):

    def test_linear_penn_convolution_same_as_isogcn(self):
        penn_ = penn.PENN(setting.BlockSetting(
            nodes=[5, 10], bias=False, support_input_indices=[0, 1, 2, 3],
            optional={
                'propagations': ['convolution'],
                'use_mlp': True,
            }))
        iso_gcn_ = iso_gcn.IsoGCN(setting.BlockSetting(
            nodes=[5, 10], support_input_indices=[0, 1, 2], optional={
                'propagations': ['convolution'],
                'create_subchain': False}))

        data_path = Path(
            'tests/data/heat_boundary/preprocessed/cylinder/clscale0.3/'
            'steepness1.0_rep0')
        np_minv = np.load(data_path / 'inversed_moment_tensors_1.npy')

        inc_grad_x = sp.load_npz(data_path / 'inc_grad_x.npz')
        inc_grad_y = sp.load_npz(data_path / 'inc_grad_y.npz')
        inc_grad_z = sp.load_npz(data_path / 'inc_grad_z.npz')
        inc_int = sp.load_npz(data_path / 'inc_int.npz')

        penn_supports = datasets.convert_sparse_tensor([
            datasets.pad_sparse(inc_grad_x),
            datasets.pad_sparse(inc_grad_y),
            datasets.pad_sparse(inc_grad_z),
            datasets.pad_sparse(inc_int)])

        gx = sp.load_npz(data_path / 'nodal_grad_x_1.npz')
        gy = sp.load_npz(data_path / 'nodal_grad_y_1.npz')
        gz = sp.load_npz(data_path / 'nodal_grad_z_1.npz')
        iso_gcn_supports = datasets.convert_sparse_tensor([
            datasets.pad_sparse(gx),
            datasets.pad_sparse(gy),
            datasets.pad_sparse(gz)])
        n = gx.shape[-1]
        np_phi = np.random.rand(n, 5).astype(np.float32)

        res_penn = penn_(
            torch.from_numpy(np_phi), torch.from_numpy(np_minv),
            supports=penn_supports)
        res_isogcn = penn_.mlp(iso_gcn_(
            torch.from_numpy(np_phi), supports=iso_gcn_supports))

        np.testing.assert_almost_equal(
            res_penn.detach().numpy(), res_isogcn.detach().numpy(), decimal=6)

    def test_linear_penn_tensor_product_same_as_isogcn(self):
        penn_ = penn.PENN(setting.BlockSetting(
            nodes=[5, 10], bias=False, support_input_indices=[0, 1, 2, 3],
            optional={
                'propagations': ['tensor_product'],
                'use_mlp': True,
            }))
        iso_gcn_ = iso_gcn.IsoGCN(setting.BlockSetting(
            nodes=[5, 10], support_input_indices=[0, 1, 2], optional={
                'propagations': ['tensor_product'],
                'create_subchain': False}))

        data_path = Path(
            'tests/data/heat_boundary/preprocessed/cylinder/clscale0.3/'
            'steepness1.0_rep0')
        np_minv = np.load(data_path / 'inversed_moment_tensors_1.npy')

        inc_grad_x = sp.load_npz(data_path / 'inc_grad_x.npz')
        inc_grad_y = sp.load_npz(data_path / 'inc_grad_y.npz')
        inc_grad_z = sp.load_npz(data_path / 'inc_grad_z.npz')
        inc_int = sp.load_npz(data_path / 'inc_int.npz')

        penn_supports = datasets.convert_sparse_tensor([
            datasets.pad_sparse(inc_grad_x),
            datasets.pad_sparse(inc_grad_y),
            datasets.pad_sparse(inc_grad_z),
            datasets.pad_sparse(inc_int)])

        gx = sp.load_npz(data_path / 'nodal_grad_x_1.npz')
        gy = sp.load_npz(data_path / 'nodal_grad_y_1.npz')
        gz = sp.load_npz(data_path / 'nodal_grad_z_1.npz')
        iso_gcn_supports = datasets.convert_sparse_tensor([
            datasets.pad_sparse(gx),
            datasets.pad_sparse(gy),
            datasets.pad_sparse(gz)])
        n = gx.shape[-1]
        np_v = np.random.rand(n, 3, 5).astype(np.float32)

        res_penn = penn_(
            torch.from_numpy(np_v), torch.from_numpy(np_minv),
            supports=penn_supports)
        res_isogcn = penn_.mlp(iso_gcn_(
            torch.from_numpy(np_v), supports=iso_gcn_supports))

        np.testing.assert_almost_equal(
            res_penn.detach().numpy(), res_isogcn.detach().numpy(), decimal=6)

    def test_linear_penn_contraction_same_as_isogcn(self):
        penn_ = penn.PENN(setting.BlockSetting(
            nodes=[5, 10], bias=False, support_input_indices=[0, 1, 2, 3],
            optional={
                'propagations': ['contraction'],
                'use_mlp': True,
            }))
        iso_gcn_ = iso_gcn.IsoGCN(setting.BlockSetting(
            nodes=[5, 10], support_input_indices=[0, 1, 2], optional={
                'propagations': ['contraction'],
                'create_subchain': False}))

        data_path = Path(
            'tests/data/heat_boundary/preprocessed/cylinder/clscale0.3/'
            'steepness1.0_rep0')
        np_minv = np.load(data_path / 'inversed_moment_tensors_1.npy')

        inc_grad_x = sp.load_npz(data_path / 'inc_grad_x.npz')
        inc_grad_y = sp.load_npz(data_path / 'inc_grad_y.npz')
        inc_grad_z = sp.load_npz(data_path / 'inc_grad_z.npz')
        inc_int = sp.load_npz(data_path / 'inc_int.npz')

        penn_supports = datasets.convert_sparse_tensor([
            datasets.pad_sparse(inc_grad_x),
            datasets.pad_sparse(inc_grad_y),
            datasets.pad_sparse(inc_grad_z),
            datasets.pad_sparse(inc_int)])

        gx = sp.load_npz(data_path / 'nodal_grad_x_1.npz')
        gy = sp.load_npz(data_path / 'nodal_grad_y_1.npz')
        gz = sp.load_npz(data_path / 'nodal_grad_z_1.npz')
        iso_gcn_supports = datasets.convert_sparse_tensor([
            datasets.pad_sparse(gx),
            datasets.pad_sparse(gy),
            datasets.pad_sparse(gz)])

        n = gx.shape[-1]
        np_v = np.random.rand(n, 3, 3, 5).astype(np.float32)

        res_penn = penn_(
            torch.from_numpy(np_v), torch.from_numpy(np_minv),
            supports=penn_supports)
        res_isogcn = penn_.mlp(iso_gcn_(
            torch.from_numpy(np_v), supports=iso_gcn_supports))

        np.testing.assert_almost_equal(
            res_penn.detach().numpy(), res_isogcn.detach().numpy(), decimal=6)

    def test_linear_penn_rotation_same_as_isogcn(self):
        penn_ = penn.PENN(setting.BlockSetting(
            nodes=[1, 1], bias=False, support_input_indices=[0, 1, 2, 3],
            optional={
                'propagations': ['rotation'],
                'use_mlp': True,
            }))
        iso_gcn_ = iso_gcn.IsoGCN(setting.BlockSetting(
            nodes=[5, 10], support_input_indices=[0, 1, 2], optional={
                'propagations': ['rotation'],
                'create_subchain': False}))

        data_path = Path(
            'tests/data/heat_boundary/preprocessed/cylinder/clscale0.3/'
            'steepness1.0_rep0')
        np_minv = np.load(data_path / 'inversed_moment_tensors_1.npy')

        inc_grad_x = sp.load_npz(data_path / 'inc_grad_x.npz')
        inc_grad_y = sp.load_npz(data_path / 'inc_grad_y.npz')
        inc_grad_z = sp.load_npz(data_path / 'inc_grad_z.npz')
        inc_int = sp.load_npz(data_path / 'inc_int.npz')

        penn_supports = datasets.convert_sparse_tensor([
            datasets.pad_sparse(inc_grad_x),
            datasets.pad_sparse(inc_grad_y),
            datasets.pad_sparse(inc_grad_z),
            datasets.pad_sparse(inc_int)])

        gx = sp.load_npz(data_path / 'nodal_grad_x_1.npz')
        gy = sp.load_npz(data_path / 'nodal_grad_y_1.npz')
        gz = sp.load_npz(data_path / 'nodal_grad_z_1.npz')
        iso_gcn_supports = datasets.convert_sparse_tensor([
            datasets.pad_sparse(gx),
            datasets.pad_sparse(gy),
            datasets.pad_sparse(gz)])

        n = gx.shape[-1]
        np_v = np.random.rand(n, 3, 1).astype(np.float32)

        res_penn = penn_(
            torch.from_numpy(np_v), torch.from_numpy(np_minv),
            supports=penn_supports)
        res_isogcn = penn_.mlp(iso_gcn_(
            torch.from_numpy(np_v), supports=iso_gcn_supports))

        np.testing.assert_almost_equal(
            res_penn.detach().numpy(), res_isogcn.detach().numpy(), decimal=6)

    def test_penn_rotation_thermal_stress_rank0_rank2(self):
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress/penn_rank0_rank2.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        tr.train()

        # Confirm inference result has isometric invariance and equivariance
        original_path = Path(
            'tests/data/rotation_thermal_stress/preprocessed/cube/original')
        transformed_paths = self.collect_transformed_paths(
            'tests/data/rotation_thermal_stress/preprocessed/cube'
            '/*_transformed_*')
        ir = inferer.Inferer(
            main_setting, save=False,
            converter_parameters_pkl=Path(
                'tests/data/rotation_thermal_stress/preprocessed'
                '/preprocessors.pkl'))
        model_directory = str(main_setting.trainer.output_directory)
        original_results = ir.infer(
            model=model_directory,
            data_directories=[original_path])
        transformed_results = ir.infer(
            model=model_directory,
            data_directories=transformed_paths)

        self.validate_results(
            original_results, transformed_results, rank2='nodal_strain_mat',
            threshold_percent=.002, decimal=4)

    def test_iso_gcn_rotation_thermal_stress_rank2_rank0(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation_thermal_stress/penn_rank2_rank0.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, .5)

        # Confirm inference result has isometric invariance
        original_path = Path(
            'tests/data/rotation_thermal_stress/preprocessed/cube/original')
        transformed_paths = self.collect_transformed_paths(
            'tests/data/rotation_thermal_stress/preprocessed/cube'
            '/*_transformed_*')
        ir = inferer.Inferer(
            main_setting, save=False,
            converter_parameters_pkl=Path(
                'tests/data/rotation_thermal_stress/preprocessed/'
                'preprocessors.pkl'))
        model_directory = str(main_setting.trainer.output_directory)
        inference_outpout_directory = \
            main_setting.trainer.output_directory / 'inferred'
        if inference_outpout_directory.exists():
            shutil.rmtree(inference_outpout_directory)
        original_results = ir.infer(
            model=model_directory,
            data_directories=[original_path])
        transformed_results = ir.infer(
            model=model_directory,
            data_directories=transformed_paths)

        self.validate_results(
            original_results, transformed_results, rank0='initial_temperature',
            decimal=4)

    def test_iso_gcn_rotation_thermal_stress_rank2_rank2(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation_thermal_stress/penn_rank2_rank2.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

        # Confirm inference result has isometric invariance
        original_path = Path(
            'tests/data/rotation_thermal_stress/preprocessed/cube/original')
        transformed_paths = self.collect_transformed_paths(
            'tests/data/rotation_thermal_stress/preprocessed/cube'
            '/*_transformed_*')
        ir = inferer.Inferer(
            main_setting, save=False,
            converter_parameters_pkl=Path(
                'tests/data/rotation_thermal_stress/preprocessed/'
                'preprocessors.pkl'))
        model_directory = str(main_setting.trainer.output_directory)
        inference_outpout_directory = \
            main_setting.trainer.output_directory / 'inferred'
        if inference_outpout_directory.exists():
            shutil.rmtree(inference_outpout_directory)
        original_results = ir.infer(
            model=model_directory,
            data_directories=[original_path])
        transformed_results = ir.infer(
            model=model_directory,
            data_directories=transformed_paths)

        self.validate_results(
            original_results, transformed_results, rank2='nodal_strain_mat',
            threshold_percent=.002, decimal=4)

    def test_grad_neumann_equivariance(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/grad/penn.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, .5)

        # Test equivariance
        ir = inferer.Inferer(
            main_setting,
            conversion_function=preprocess.ConversionFunctionGrad(),
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

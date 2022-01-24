
import glob
from pathlib import Path
import shutil
import unittest

import numpy as np
import scipy.sparse as sp
import torch

import siml.datasets as datasets
import siml.inferer as inferer
import siml.iso_gcn as iso_gcn
import siml.networks.penn as penn
import siml.setting as setting
import siml.trainer as trainer


PLOT = False


class TestPENN(unittest.TestCase):

    def test_linear_penn_convolution_same_as_isogcn(self):
        penn_ = penn.PENN(setting.BlockSetting(
            nodes=[1, 10], optional={
                'propagations': ['convolution']}))
        iso_gcn_ = iso_gcn.IsoGCN(setting.BlockSetting(
            nodes=[1, 10], optional={
                'propagations': ['convolution'],
                'create_subchain': False}))

        data_path = Path(
            'tests/data/heat_boundary/preprocessed/cylinder/clscale0.3/'
            'steepness1.0_rep0')
        np_phi = np.load(data_path / 'nodal_t_0.npy')

        inc_grad = sp.load_npz(data_path / 'inc_grad.npz')
        inc_int_x = sp.load_npz(data_path / 'inc_int_x.npz')
        inc_int_y = sp.load_npz(data_path / 'inc_int_y.npz')
        inc_int_z = sp.load_npz(data_path / 'inc_int_z.npz')

        penn_supports = datasets.convert_sparse_tensor([
            datasets.pad_sparse(inc_grad),
            datasets.pad_sparse(inc_int_x),
            datasets.pad_sparse(inc_int_y),
            datasets.pad_sparse(inc_int_z)])

        gx = sp.load_npz(data_path / 'nodal_grad_x_1.npz')
        gy = sp.load_npz(data_path / 'nodal_grad_y_1.npz')
        gz = sp.load_npz(data_path / 'nodal_grad_z_1.npz')
        iso_gcn_supports = datasets.convert_sparse_tensor([
            datasets.pad_sparse(gx),
            datasets.pad_sparse(gy),
            datasets.pad_sparse(gz)])

        res_penn = penn_(torch.from_numpy(np_phi), penn_supports)
        res_isogcn = iso_gcn_(torch.from_numpy(np_phi, iso_gcn_supports)

        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress/iso_gcn_frame_rank2.yml'))
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
            threshold_percent=.002)

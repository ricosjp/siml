
from pathlib import Path
import unittest

import numpy as np
import scipy.sparse as sp
import torch

import siml.datasets as datasets
import siml.networks.iso_gcn as iso_gcn
import siml.networks.penn as penn
import siml.setting as setting


PLOT = False


class TestPENN(unittest.TestCase):

    def test_linear_penn_convolution_same_as_isogcn(self):
        penn_ = penn.PENN(setting.BlockSetting(
            nodes=[1, 10], bias=False, support_input_indices=[0, 1, 2, 3],
            optional={
                'propagations': ['convolution'],
                'use_mlp': True,
            }))
        iso_gcn_ = iso_gcn.IsoGCN(setting.BlockSetting(
            nodes=[1, 10], support_input_indices=[0, 1, 2], optional={
                'propagations': ['convolution'],
                'create_subchain': False}))

        data_path = Path(
            'tests/data/heat_boundary/preprocessed/cylinder/clscale0.3/'
            'steepness1.0_rep0')
        np_phi = np.load(data_path / 'nodal_t_0.npy')
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

        res_penn = penn_(
            torch.from_numpy(np_phi), torch.from_numpy(np_minv),
            supports=penn_supports)
        res_isogcn = penn_.mlp(iso_gcn_(
            torch.from_numpy(np_phi), supports=iso_gcn_supports))

        np.testing.assert_almost_equal(
            res_penn.detach().numpy(), res_isogcn.detach().numpy(), decimal=6)

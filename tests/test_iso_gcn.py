
import glob
from pathlib import Path
import shutil
import unittest

import numpy as np
import scipy.sparse as sp
import torch

import siml.datasets as datasets
import siml.inferer as inferer
import siml.networks.iso_gcn as iso_gcn
import siml.setting as setting
import siml.trainer as trainer


PLOT = False


class TestIsoGCN(unittest.TestCase):

    def test_convolution_rank0_rank1(self):
        x = np.random.rand(4, 3)
        h = np.random.rand(4, 2)
        rotation_matrix = self.generate_rotation_matrix()
        ig = self.generate_isogcn({
            'propagations': ['convolution'],
            'create_subchain': False,
            'symmetric': False})
        self.trial(
            x, rotation_matrix, h,
            rotate_y=self.transform_rank1, iso_gcn_=ig,
            einstring='ijk,jf->ikf')

    def test_convolution_rank0_rank2(self):
        x = np.random.rand(4, 3)
        h = np.random.rand(4, 2)
        rotation_matrix = self.generate_rotation_matrix()
        ig = self.generate_isogcn({
            'propagations': ['convolution', 'tensor_product'],
            'create_subchain': False,
            'symmetric': True})
        self.trial(
            x, rotation_matrix, h,
            rotate_y=self.transform_rank2, iso_gcn_=ig, check_symmetric=True)

    def test_contraction_rank1_rank0(self):
        x = np.random.rand(4, 3)
        h = np.random.rand(4, 3, 2)
        rotation_matrix = self.generate_rotation_matrix()
        ig = self.generate_isogcn({
            'propagations': ['contraction'], 'create_subchain': False})
        self.trial(
            x, rotation_matrix, h, rotate_x=self.transform_rank1,
            rotate_y=self.identity, iso_gcn_=ig, einstring='ijk,jkf->if')

    def test_contraction_rank2_rank1(self):
        x = np.random.rand(4, 3)
        h = np.random.rand(4, 3, 3, 2)
        rotation_matrix = self.generate_rotation_matrix()
        ig = self.generate_isogcn(optional={
            'propagations': ['contraction'], 'create_subchain': False})
        self.trial(
            x, rotation_matrix, h, rotate_x=self.transform_rank2,
            rotate_y=self.transform_rank1, iso_gcn_=ig,
            einstring='ijk,jlkf->ilf')

    def test_contraction_rank2_rank0(self):
        x = np.random.rand(4, 3)
        h = np.random.rand(4, 3, 3, 2)
        rotation_matrix = self.generate_rotation_matrix()
        ig = self.generate_isogcn({
            'propagations': ['contraction', 'contraction'],
            'create_subchain': False})
        self.trial(
            x, rotation_matrix, h, rotate_x=self.transform_rank2,
            rotate_y=self.identity, iso_gcn_=ig)

    def test_contraction_g2_rank2_rank2(self):
        x = np.random.rand(4, 3)
        h = np.random.rand(4, 3, 3, 2)
        rotation_matrix = self.generate_rotation_matrix()
        ig = self.generate_isogcn(optional={
            'propagations': ['contraction'], 'create_subchain': False,
            'support_tensor_rank': 2})
        self.trial(
            x, rotation_matrix, h, rotate_x=self.transform_rank2,
            rotate_y=self.transform_rank2, iso_gcn_=ig, rank_g=2,
            einstring='ijkm,jlmf->iklf')

    def test_convolution_g2_rank0_rank2(self):
        x = np.random.rand(4, 3)
        h = np.random.rand(4, 2)
        rotation_matrix = self.generate_rotation_matrix()
        ig = self.generate_isogcn(optional={
            'propagations': ['convolution'], 'create_subchain': False,
            'support_tensor_rank': 2})
        self.trial(
            x, rotation_matrix, h, rotate_x=self.identity,
            rotate_y=self.transform_rank2, iso_gcn_=ig, rank_g=2,
            einstring='ijkm,jf->ikmf')

    def test_convolution_g2_rank0_rank2_symmetric(self):
        x = np.random.rand(4, 3)
        h = np.random.rand(4, 2)
        rotation_matrix = self.generate_rotation_matrix()
        ig = self.generate_isogcn(optional={
            'propagations': ['convolution'], 'create_subchain': False,
            'support_tensor_rank': 2, 'symmetric': True})
        self.trial(
            x, rotation_matrix, h, rotate_x=self.identity,
            rotate_y=self.transform_rank2, iso_gcn_=ig, rank_g=2,
            einstring='ijkm,jf->ikmf', check_symmetric=True)

    def trial(
            self, x, rotation_matrix, h,
            *, rotate_x=None, rotate_y=None, iso_gcn_=None,
            check_symmetric=False, rank_g=1, einstring=None):
        """Operate IsoGCN Layer and confirm its invariance or equivariance.

        Parameters
        ----------
        x: np.ndarray
            (n_node, dim) shaped array of vertex positions.
        h: np.ndarray
            (n_node, dim, dim, ..., n_feature) shaped array of a collection
                    ^^^^^^^^^^^^^^^
                    k repetition for rank k tensors
            of input tensors.
        rotate_x: callable, optional
            Callable to rotate the input. For identity, input None. The default
            is None.
        rotate_y: callable, optional
            Callable to rotate the output. For identity, input None.
            The default is None.
        iso_gcn: siml.networks.IsoGCN, optional
            IsoGCN layer. If not fed, created automatically with the default
            setting.
        check_symmetric: bool, optional
            If True, check wheather the output is symmetric. The default is
            False.
        """
        _, _, g_tilde = self.generate_gs(x)
        original_h_conv = self.conv(
            iso_gcn_, h, g_tilde, rank_g=rank_g, einstring=einstring)

        rotated_x = self.transform_rank1(rotation_matrix, x)
        _, _, rotated_g_tilde = self.generate_gs(rotated_x)
        if rotate_x is None:
            rotated_h_conv = self.conv(
                iso_gcn_, h, rotated_g_tilde, rank_g=rank_g)
        else:
            rotated_h_conv = self.conv(
                iso_gcn_, rotate_x(rotation_matrix, h), rotated_g_tilde,
                rank_g=rank_g)
        print('Rotation matrix:')
        print(rotation_matrix)
        self.print_vec(rotated_h_conv, 'IsoGCN x rotation')

        original_rotated_h_conv = rotate_y(rotation_matrix, original_h_conv)
        self.print_vec(original_rotated_h_conv, 'rotation x IsoGCN')

        np.testing.assert_array_almost_equal(
            rotated_h_conv, original_rotated_h_conv)

        if check_symmetric:
            for i in range(h.shape[-1]):
                np.testing.assert_array_almost_equal(
                    original_rotated_h_conv[:, 0, 1, i],
                    original_rotated_h_conv[:, 1, 0, i])
                np.testing.assert_array_almost_equal(
                    original_rotated_h_conv[:, 0, 2, i],
                    original_rotated_h_conv[:, 2, 0, i])
                np.testing.assert_array_almost_equal(
                    original_rotated_h_conv[:, 1, 2, i],
                    original_rotated_h_conv[:, 2, 1, i])
        return

    def collect_transformed_paths(self, root_path, recursive=False):
        return [
            Path(g) for g in glob.glob(str(root_path), recursive=recursive)]

    def load_orthogonal_matrix(self, preprocessed_path):
        return np.loadtxt(
            str(preprocessed_path / 'orthogonal_matrix.txt').replace(
                'preprocessed', 'raw'))

    def test_contraction_rank2_rank0_real_data(self):
        original_path = Path(
            'tests/data/rotation_thermal_stress/preprocessed/cube/original')
        rotated_path = Path(
            'tests/data/rotation_thermal_stress/preprocessed/cube'
            '/original_transformed_rotation_yz')
        rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        mirrored_path = Path(
            'tests/data/rotation_thermal_stress/preprocessed/cube'
            '/original_transformed_mirror_xy')
        mirror_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

        ig = self.generate_isogcn({
            'propagations': ['contraction', 'contraction'],
            'create_subchain': False})
        h = np.load(original_path / 'nodal_strain_mat.npy')
        rotated_h = self.transform_rank2(rotation_matrix, h).astype(np.float32)
        mirrored_h = self.transform_rank2(mirror_matrix, h).astype(np.float32)

        # Use dense
        original_genam = self.load_genam(original_path)
        rotated_genam = self.load_genam(rotated_path)
        original_conv = self.conv(ig, h, original_genam)
        rotated_conv = self.conv(ig, rotated_h, rotated_genam)
        self.print_vec(original_conv, 'rotation x IsoGCN', 10)
        self.print_vec(rotated_conv, 'IsoGCN x rotation', 10)
        self.print_vec(rotated_conv - original_conv, 'Diff', 10)
        np.testing.assert_array_almost_equal(
            rotated_conv / 100, original_conv / 100, decimal=5)

        # Compare with hand written results
        hess = np.einsum(
            'ijp,jkq->ikpq',
            original_genam[:10, :10], original_genam[:10, :10])
        siml_conv = self.conv(ig, h[:10] / 100, original_genam[:10, :10])
        handwritten_original_conv = np.einsum(
            'ijpq,jpqf->if', hess, h[:10] / 100)
        self.print_vec(siml_conv, 'IsoGCN', 10)
        self.print_vec(handwritten_original_conv, 'einsum', 10)
        self.print_vec(siml_conv - handwritten_original_conv, 'Diff', 10)
        np.testing.assert_almost_equal(
            siml_conv, handwritten_original_conv, decimal=5)

        # Use sparse
        torch_original_genam = self.load_genam(original_path, mode='torch')
        torch_rotated_genam = self.load_genam(rotated_path, mode='torch')
        torch_mirrored_genam = self.load_genam(mirrored_path, mode='torch')

        torch_original_conv = ig(
            torch.from_numpy(h), torch_original_genam).numpy()

        torch_rotated_conv = ig(
            torch.from_numpy(rotated_h), torch_rotated_genam).numpy()
        self.print_vec(torch_original_conv, 'rotation x IsoGCN', 10)
        self.print_vec(torch_rotated_conv, 'IsoGCN x rotation', 10)
        self.print_vec(
            torch_rotated_conv - torch_original_conv, 'Diff', 10)
        np.testing.assert_array_almost_equal(
            torch_original_conv / 100, torch_rotated_conv / 100, decimal=5)

        torch_mirrored_conv = ig(
            torch.from_numpy(mirrored_h), torch_mirrored_genam).numpy()
        self.print_vec(torch_original_conv, 'mirror x IsoGCN', 10)
        self.print_vec(torch_mirrored_conv, 'IsoGCN x mirror', 10)
        self.print_vec(
            torch_mirrored_conv - torch_original_conv, 'Diff', 10)
        np.testing.assert_array_almost_equal(
            torch_original_conv / 100, torch_mirrored_conv / 100, decimal=5)
        return

    def test_convolution_rank0_rank2_real_data(self):
        original_path = Path(
            'tests/data/rotation_thermal_stress/preprocessed/cube/original')
        rotated_path = Path(
            'tests/data/rotation_thermal_stress/preprocessed/cube'
            '/original_transformed_rotation_yz')
        rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        mirrored_path = Path(
            'tests/data/rotation_thermal_stress/preprocessed/cube'
            '/original_transformed_mirror_xy')
        mirror_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

        ig = self.generate_isogcn({
            'propagations': ['convolution', 'tensor_product'],
            'create_subchain': False,
            'symmetric': False})
        h = np.load(original_path / 'initial_temperature.npy')

        # Use dense
        original_genam = self.load_genam(original_path)
        rotated_genam = self.load_genam(rotated_path)
        original_conv = self.conv(ig, h, original_genam)
        rotated_original_conv = self.transform_rank2(
            rotation_matrix, original_conv)
        rotated_conv = self.conv(ig, h, rotated_genam)
        self.print_vec(rotated_original_conv, 'rotation x IsoGCN', 10)
        self.print_vec(rotated_conv, 'IsoGCN x rotation', 10)
        self.print_vec(rotated_conv - rotated_original_conv, 'Diff', 10)
        np.testing.assert_array_almost_equal(
            rotated_conv, rotated_original_conv)

        # Compare with hand written results
        hess = np.einsum(
            'ijp,jkq->ikpq',
            original_genam[:10, :10], original_genam[:10, :10])
        siml_conv = self.conv(ig, h[:10] / 100, original_genam[:10, :10])
        handwritten_original_conv = np.einsum(
            'ijpq,jf->ipqf', hess, h[:10] / 100)
        self.print_vec(siml_conv, 'IsoGCN', 5)
        self.print_vec(handwritten_original_conv, 'einsum', 5)
        self.print_vec(siml_conv - handwritten_original_conv, 'Diff', 5)
        np.testing.assert_almost_equal(
            siml_conv, handwritten_original_conv, decimal=6)

        # Use sparse
        torch_original_genam = self.load_genam(original_path, mode='torch')
        torch_rotated_genam = self.load_genam(rotated_path, mode='torch')
        torch_mirrored_genam = self.load_genam(mirrored_path, mode='torch')

        torch_original_conv = ig(
            torch.from_numpy(h), torch_original_genam).numpy()

        torch_rotated_original_conv = self.transform_rank2(
            rotation_matrix, torch_original_conv)
        torch_rotated_conv = ig(
            torch.from_numpy(h), torch_rotated_genam).numpy()
        self.print_vec(torch_rotated_original_conv, 'rotation x IsoGCN', 10)
        self.print_vec(torch_rotated_conv, 'IsoGCN x rotation', 10)
        self.print_vec(
            torch_rotated_conv - torch_rotated_original_conv, 'Diff', 10)
        np.testing.assert_array_almost_equal(
            torch_rotated_original_conv, torch_rotated_conv)

        torch_mirrored_original_conv = self.transform_rank2(
            mirror_matrix, torch_original_conv)
        torch_mirrored_conv = ig(
            torch.from_numpy(h), torch_mirrored_genam).numpy()
        self.print_vec(torch_mirrored_original_conv, 'mirror x IsoGCN', 10)
        self.print_vec(torch_mirrored_conv, 'IsoGCN x mirror', 10)
        self.print_vec(
            torch_mirrored_conv - torch_mirrored_original_conv, 'Diff', 10)
        np.testing.assert_array_almost_equal(
            torch_mirrored_original_conv, torch_mirrored_conv)
        return

    def load_genam(self, data_path, mode='numpy'):
        gx = sp.load_npz(data_path / 'nodal_grad_x.npz')
        gy = sp.load_npz(data_path / 'nodal_grad_y.npz')
        gz = sp.load_npz(data_path / 'nodal_grad_z.npz')
        if mode == 'numpy':
            return np.stack([
                gx.toarray(), gy.toarray(), gz.toarray()], axis=-1)
        elif mode == 'scipy':
            return [gx, gy, gz]
        elif mode == 'torch':
            return datasets.convert_sparse_tensor([self.load_genam(
                data_path, mode='sparse_info')])[0]
        elif mode == 'sparse_info':
            return [datasets.pad_sparse(g) for g in [gx, gy, gz]]
        else:
            raise ValueError(f"Unexpected mode: {mode}")

    def generate_rotation_matrix(self):
        def normalize(x):
            return x / np.linalg.norm(x)

        vec1 = normalize(np.random.rand(3)*2 - 1)
        vec2 = normalize(np.random.rand(3)*2 - 1)
        vec3 = normalize(np.cross(vec1, vec2))
        vec2 = np.cross(vec3, vec1)
        return np.array([vec1, vec2, vec3])

    def print_vec(self, x, name=None, n=None):
        if n is None:
            n = x.shape[0]
        print('--')
        if name is not None:
            print(name)
        if len(x.shape) == 4:
            for _x in x[:n, ..., 0]:
                print(_x)
        elif len(x.shape) == 3:
            for _x in x[:n, ..., 0]:
                print(_x)
        elif len(x.shape) == 2:
            for _x in x[:n]:
                print(_x)
        else:
            raise ValueError(f"Unexpected array shape: {x.shape}")
        print('--')
        return

    def transform_rank1(self, orthogonal_matrix, x):
        if len(x.shape) == 2:
            return np.array([orthogonal_matrix @ _x for _x in x])
        elif len(x.shape) == 3:
            n_feature = x.shape[-1]
            return np.stack([
                np.array([orthogonal_matrix @ _x for _x in x[..., i_feature]])
                for i_feature in range(n_feature)], axis=-1)
        else:
            raise ValueError(f"Unexpected x shape: {x.shape}")

    def identity(self, orthogonal_matrix, x):
        return x

    def transform_rank2(self, orthogonal_matrix, t):
        n_feature = t.shape[-1]
        return np.stack([
            np.array([
                orthogonal_matrix @ _t @ orthogonal_matrix.T
                for _t in t[..., i_feature]])
            for i_feature in range(n_feature)], axis=-1)

    def conv(self, iso_gcn_, h, g_tilde, rank_g=1, einstring=None):
        if rank_g == 1:
            gs = [
                torch.from_numpy(g_tilde[..., i])
                for i in range(g_tilde.shape[-1])]
        elif rank_g == 2:
            g_tilde = np.einsum('ijk,ijl->ijkl', g_tilde, g_tilde)
            gs = [
                torch.from_numpy(g_tilde[..., i_row, i_col])
                for i_row in range(g_tilde.shape[-1])
                for i_col in range(g_tilde.shape[-1])]
        else:
            raise NotImplementedError

        torch_res = iso_gcn_._forward_single(torch.from_numpy(h), gs).numpy()
        if einstring is not None:
            np.testing.assert_almost_equal(
                torch_res, np.einsum(einstring, g_tilde, h))
        return torch_res

    def generate_isogcn(self, optional={}, bias=True):
        block_setting = setting.BlockSetting(
            type='iso_gcn', residual=False, support_input_indices=[0, 1, 2],
            nodes=[-1, -1], activations=['identity'], optional=optional,
            bias=bias)
        return iso_gcn.IsoGCN(block_setting)

    def generate_gs(self, x):
        n = len(x)
        g = np.array([
            [x[col] - x[row] for col in range(n)] for row in range(n)])
        g_eye = np.einsum('ij,ik->ijk', np.eye(n), np.sum(g, axis=1))
        g_tilde = g - g_eye
        return g, g_eye, g_tilde

    def validate_results(
            self, original_results, transformed_results,
            *, rank0=None, rank2=None, validate_x=True, decimal=5,
            threshold_percent=1e-3):

        if rank0 is not None:
            scale = np.max(np.abs(original_results[0]['dict_y'][rank0]))

            for transformed_result in transformed_results:
                if validate_x:
                    np.testing.assert_almost_equal(
                        original_results[0]['dict_x'][rank0],
                        transformed_result['dict_x'][rank0], decimal=decimal)

                print(
                    f"data_directory: {transformed_result['data_directory']}")
                self.print_vec(
                    original_results[0]['dict_y'][rank0] / scale,
                    'Transform x IsoGCN', 5)
                self.print_vec(
                    transformed_result['dict_y'][rank0] / scale,
                    'IsoGCN x Transform', 5)
                self.print_vec((
                    original_results[0]['dict_y'][rank0]
                    - transformed_result['dict_y'][rank0]) / scale,
                    'Diff', 5)
                self.compare_relative_rmse(
                    original_results[0]['dict_y'][rank0],
                    transformed_result['dict_y'][rank0],
                    threshold_percent=threshold_percent)
                np.testing.assert_almost_equal(
                    original_results[0]['dict_y'][rank0] / scale,
                    transformed_result['dict_y'][rank0] / scale,
                    decimal=decimal)

        if rank2 is not None:
            scale = np.max(np.abs(original_results[0]['dict_y'][rank2]))

            for transformed_result in transformed_results:
                print(
                    f"data_directory: {transformed_result['data_directory']}")
                orthogonal_matrix = self.load_orthogonal_matrix(
                    transformed_result['data_directory'])
                if validate_x:
                    np.testing.assert_almost_equal(
                        self.transform_rank2(
                            orthogonal_matrix,
                            original_results[0]['dict_x'][rank2]),
                        transformed_result['dict_x'][rank2], decimal=decimal)

                transformed_original = self.transform_rank2(
                    orthogonal_matrix, original_results[0]['dict_y'][rank2])
                self.print_vec(
                    transformed_original / scale, 'Transform x IsoGCN', 5)
                self.print_vec(
                    transformed_result['dict_y'][rank2] / scale,
                    'IsoGCN x Transform', 5)
                self.print_vec((
                    transformed_original
                    - transformed_result['dict_y'][rank2]) / scale,
                    'Diff', 5)
                self.compare_relative_rmse(
                    transformed_original, transformed_result['dict_y'][rank2],
                    threshold_percent=threshold_percent)
                np.testing.assert_almost_equal(
                    transformed_original / scale,
                    transformed_result['dict_y'][rank2] / scale,
                    decimal=decimal)

        return

    def compare_relative_rmse(self, target, y, threshold_percent):
        target_scale = np.mean(target**2)**.5
        rmse = np.mean((y - target)**2)**.5
        self.assertLess(rmse / target_scale * 100, threshold_percent)

    def test_iso_gcn_rank0_rank0(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation/iso_gcn_rank0_rank0.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)
        self.assertEqual(len(tr.model.dict_block['ISO_GCN1'].subchains), 1)
        self.assertEqual(len(tr.model.dict_block['ISO_GCN2'].subchains), 1)

        # Confirm results does not change under rigid body transformation
        original_path = Path(
            'tests/data/rotation/preprocessed/cube/clscale1.0/original')
        transformed_paths = self.collect_transformed_paths(
            'tests/data/rotation/preprocessed/cube/clscale1.0/rotated')
        ir = inferer.Inferer(
            main_setting, save=False,
            converter_parameters_pkl=Path(
                'tests/data/rotation/preprocessed/preprocessors.pkl'))
        model_directory = str(main_setting.trainer.output_directory)
        original_results = ir.infer(
            model=model_directory,
            data_directories=[original_path])
        transformed_results = ir.infer(
            model=model_directory,
            data_directories=transformed_paths)

        self.validate_results(
            original_results, transformed_results, rank0='t_100')

    def test_iso_gcn_rank0_rank0_implicit(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation/iso_gcn_rank0_rank0_implicit.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 1.)

        # Confirm results does not change under rigid body transformation
        original_path = Path(
            'tests/data/rotation/preprocessed/cube/clscale1.0/original')
        transformed_paths = self.collect_transformed_paths(
            'tests/data/rotation/preprocessed/cube/clscale1.0/rotated')
        ir = inferer.Inferer(
            main_setting, save=False,
            converter_parameters_pkl=Path(
                'tests/data/rotation/preprocessed/preprocessors.pkl'))
        model_directory = str(main_setting.trainer.output_directory)
        original_results = ir.infer(
            model=model_directory,
            data_directories=[original_path])
        transformed_results = ir.infer(
            model=model_directory,
            data_directories=transformed_paths)

        self.validate_results(
            original_results, transformed_results, rank0='t_100')

    def test_iso_gcn_inverse_temperature(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/heat_time_series/iso_gcn_inverse.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 5.)

    def test_iso_gcn_rotation_thermal_stress_rank2_rank0(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation_thermal_stress/iso_gcn_rank2_rank0.yml'))
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
            decimal=5)

    def test_iso_gcn_inverse_invariant(self):
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress'
            '/iso_gcn_rank0_rank0_inverse.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 5.)

        # Confirm inference result has isometric invariance
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

        # NOTE: LTE is not invariant, but just to validate IsoGCN invariance,
        #       we use it.
        self.validate_results(
            original_results, transformed_results, rank0='global_lte_array',
            validate_x=False)

    def call_model(self, model, h, genam):
        gs = [[
            torch.from_numpy(genam[..., i]) for i in range(genam.shape[-1])]]
        return model({'x': torch.from_numpy(h), 'supports': gs}).numpy()

    def test_iso_gcn_rotation_thermal_stress_rank0_rank2_wo_postprocess(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation_thermal_stress/iso_gcn_rank0_rank2.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        tr.train()

        # Confirm inference result has isometric equivariance
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
            decimal=3, threshold_percent=2e-2)

    def test_iso_gcn_rotation_thermal_stress_rank0_rank2_with_postprocess(
            self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation_thermal_stress/iso_gcn_rank0_rank2.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        tr.train()

        # Confirm inference result has isometric equivariance
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
            decimal=3, threshold_percent=2e-2)

    def test_iso_gcn_rotation_thermal_stress_rank2_rank2(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation_thermal_stress/iso_gcn_rank2_rank2.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        tr.train()

        # Confirm inference result has isometric equivariance
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
            original_results, transformed_results, rank2='nodal_lte_mat',
            decimal=5)

    def test_iso_gcn_rotation_thermal_stress_rank2_rank2_nonlinear(self):
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress/'
            'iso_gcn_rank2_rank2_nonlinear.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        tr.train()

        # Confirm inference result has isometric equivariance
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
            original_results, transformed_results, rank2='nodal_lte_mat',
            decimal=5)

    def test_sgcn_rotation_thermal_stress_rank2_rank2(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation_thermal_stress/sgcn_rank2_rank2.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        tr.train()

        # Confirm inference result has isometric equivariance
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
            original_results, transformed_results, rank2='nodal_lte_mat',
            decimal=5)

    def test_iso_gcn_rotation_thermal_stress_dict_input_list_output(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation_thermal_stress/iso_gcn_dict_input.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 5.)

        # Confirm inference result has isometric equivariance
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
            threshold_percent=2., decimal=1)

    def test_iso_gcn_rotation_thermal_stress_dict_input_dict_output(self):
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress'
            '/iso_gcn_dict_input_dict_output.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        tr.train()

        # Confirm inference result has isometric equivariance
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
            original_results, transformed_results,
            rank0='cnt_temperature', rank2='global_lte_mat')

    def test_iso_gcn_rotation_thermal_stress_rank0_rank2_global_pooling(self):
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress/iso_gcn_rank0_rank2_pool.yml'))
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
            original_results, transformed_results, rank2='global_lte_mat',
            threshold_percent=.002)

    def test_iso_gcn_rotation_thermal_stress_rank0_rank2_diag(self):
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress/iso_gcn_rank0_rank2_diag.yml'))
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

    def test_iso_gcn_rotation_thermal_stress_frame_rank2(self):
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

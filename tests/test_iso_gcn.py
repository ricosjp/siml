
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


class TestNetwork(unittest.TestCase):

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
            rotate=self.rotate_rank1, iso_gcn_=ig)

    def test_convolution_rank0_rank2(self):
        x = np.random.rand(4, 3)
        h = np.random.rand(4, 2)
        rotation_matrix = self.generate_rotation_matrix()
        ig = self.generate_isogcn({
            'propagations': ['convolution', 'convolution'],
            'create_subchain': False,
            'symmetric': True})
        self.trial(
            x, rotation_matrix, h,
            rotate=self.rotate_rank2, iso_gcn_=ig)

    def trial(self, x, rotation_matrix, h, *, rotate=None, iso_gcn_=None):
        g, g_eye, g_tilde = self.generate_gs(x)
        original_h_conv = self.conv(iso_gcn_, h, g_tilde)

        rotated_x = self.rotate_rank1(rotation_matrix, x)
        _, _, rotated_g_tilde = self.generate_gs(rotated_x)
        rotated_h_conv = self.conv(iso_gcn_, h, rotated_g_tilde)
        self.print_vec(rotated_h_conv, 'IsoGCN x rotation')

        original_rotated_h_conv = rotate(rotation_matrix, original_h_conv)
        self.print_vec(original_rotated_h_conv, 'rotation x IsoGCN')

        np.testing.assert_array_almost_equal(
            rotated_h_conv, original_rotated_h_conv)
        return

    def test_convolution_rank0_rank2_real_data(self):
        original_path = Path(
            'tests/data/rotation_thermal_stress/preprocessed/cube/original')
        rotated_path = Path(
            'tests/data/rotation_thermal_stress/preprocessed/cube/rotated')
        rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        ig = self.generate_isogcn({
            'propagations': ['convolution', 'convolution'],
            'create_subchain': False,
            'symmetric': True})
        h = np.load(original_path / 'initial_temperature.npy')

        # Use dense
        original_genam = self.load_genam(original_path)
        rotated_genam = self.load_genam(rotated_path)
        original_conv = self.conv(ig, h, original_genam)
        rotated_original_conv = self.rotate_rank2(
            rotation_matrix, original_conv)
        rotated_conv = self.conv(ig, h, rotated_genam)
        self.print_vec(rotated_original_conv, 'rotation x IsoGCN', 10)
        self.print_vec(rotated_conv, 'IsoGCN x rotation', 10)
        self.print_vec(rotated_conv - rotated_original_conv, 'Diff', 10)
        np.testing.assert_array_almost_equal(
            rotated_conv, rotated_original_conv)

        # Use sparse
        torch_original_genam = self.load_genam(original_path, mode='torch')
        torch_rotated_genam = self.load_genam(rotated_path, mode='torch')
        torch_original_conv = ig(
            torch.from_numpy(h), torch_original_genam).numpy()
        torch_rotated_original_conv = self.rotate_rank2(
            rotation_matrix, torch_original_conv)
        torch_rotated_conv = ig(
            torch.from_numpy(h), torch_rotated_genam).numpy()
        self.print_vec(torch_rotated_original_conv, 'rotation x IsoGCN', 10)
        self.print_vec(torch_rotated_conv, 'IsoGCN x rotation', 10)
        self.print_vec(
            torch_rotated_conv - torch_rotated_original_conv, 'Diff', 10)

        np.testing.assert_array_almost_equal(
            torch_rotated_original_conv, torch_rotated_conv)
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
        else:
            for _x in x:
                print(_x[:n, 0])
        print('--')
        return

    def rotate_rank1(self, rotation_matrix, x):
        return np.array([rotation_matrix @ _x for _x in x])

    def rotate_rank2(self, rotation_matrix, t):
        n_feature = t.shape[-1]
        return np.stack([
            np.array([
                rotation_matrix @ _t @ rotation_matrix.T
                for _t in t[..., i_feature]])
            for i_feature in range(n_feature)], axis=-1)

    def conv(self, iso_gcn_, h, g_tilde):
        gs = [
            torch.from_numpy(g_tilde[..., i])
            for i in range(g_tilde.shape[-1])]
        return iso_gcn_(torch.from_numpy(h), gs).numpy()

    def generate_isogcn(self, optional={}):
        block_setting = setting.BlockSetting(
            type='iso_gcn', residual=False, support_input_indices=[0, 1, 2],
            nodes=[-1, -1], activations=['identity'], optional=optional)
        return iso_gcn.IsoGCN(block_setting)

    def generate_gs(self, x):
        n = len(x)
        g = np.array([
            [x[col] - x[row] for col in range(n)] for row in range(n)])
        g_eye = np.einsum('ij,ik->ijk', np.eye(n), np.sum(g, axis=1))
        g_tilde = g - g_eye
        return g, g_eye, g_tilde

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
        ir = inferer.Inferer(main_setting)
        inference_outpout_directory = \
            main_setting.trainer.output_directory / 'inferred'
        if inference_outpout_directory.exists():
            shutil.rmtree(inference_outpout_directory)
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            preprocessed_data_directory=[
                Path(
                    'tests/data/rotation/preprocessed/cube/clscale1.0/'
                    'original'),
                Path(
                    'tests/data/rotation/preprocessed/cube/clscale1.0/'
                    'rotated')],
            converter_parameters_pkl=Path(
                'tests/data/rotation/preprocessed/preprocessors.pkl'),
            output_directory=inference_outpout_directory,
            overwrite=True)
        np.testing.assert_almost_equal(
            results[0]['dict_y']['t_100'],
            results[1]['dict_y']['t_100'], decimal=5)

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

        ir = inferer.Inferer(main_setting)
        inference_outpout_directory = \
            main_setting.trainer.output_directory / 'inferred'
        if inference_outpout_directory.exists():
            shutil.rmtree(inference_outpout_directory)
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            preprocessed_data_directory=[
                Path(
                    'tests/data/rotation_thermal_stress/preprocessed/cube/'
                    'original'),
                Path(
                    'tests/data/rotation_thermal_stress/preprocessed/cube/'
                    'rotated')],
            converter_parameters_pkl=Path(
                'tests/data/rotation_thermal_stress/preprocessed/'
                'preprocessors.pkl'),
            output_directory=inference_outpout_directory,
            overwrite=True)

        # Confirm inference result has rotation invariance
        np.testing.assert_almost_equal(
            results[0]['dict_x']['initial_temperature'],
            results[1]['dict_x']['initial_temperature'])
        np.testing.assert_almost_equal(
            results[0]['dict_y']['initial_temperature'],
            results[1]['dict_y']['initial_temperature'], decimal=3)

    def test_iso_gcn_inverse_invariant(self):
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress'
            '/iso_gcn_rank0_rank0_inverse.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 5.)

        ir = inferer.Inferer(main_setting)
        inference_outpout_directory = \
            main_setting.trainer.output_directory / 'inferred'
        if inference_outpout_directory.exists():
            shutil.rmtree(inference_outpout_directory)
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            preprocessed_data_directory=[
                Path(
                    'tests/data/rotation_thermal_stress/preprocessed/cube/'
                    'original'),
                Path(
                    'tests/data/rotation_thermal_stress/preprocessed/cube/'
                    'rotated')],
            converter_parameters_pkl=Path(
                'tests/data/rotation_thermal_stress/preprocessed/'
                'preprocessors.pkl'),
            output_directory=inference_outpout_directory,
            overwrite=True)

        # Confirm inference result has rotation invariance
        # NOTE: LTE is not invariant, but just to validate IsoGCN invariance,
        #       we use it.
        print(results[0]['dict_y']['global_lte_array'])
        print(results[1]['dict_y']['global_lte_array'])
        np.testing.assert_almost_equal(
            results[0]['dict_y']['global_lte_array'],
            results[1]['dict_y']['global_lte_array'])

    def call_model(self, model, h, genam):
        gs = [[
            torch.from_numpy(genam[..., i]) for i in range(genam.shape[-1])]]
        return model({'x': torch.from_numpy(h), 'supports': gs}).numpy()

    def test_iso_gcn_rotation_thermal_stress_rank0_rank2(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation_thermal_stress/iso_gcn_rank0_rank2.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        tr.train()

        original_path = Path(
            'tests/data/rotation_thermal_stress/preprocessed/cube/original')
        rotated_path = Path(
            'tests/data/rotation_thermal_stress/preprocessed/cube/rotated')
        rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        h = np.load(original_path / 'initial_temperature.npy')
        ig = tr.model.dict_block['ISO_GCN_TO_TENSOR']

        # Use model directly
        original_genam = self.load_genam(original_path)
        rotated_genam = self.load_genam(rotated_path)
        h = np.load(original_path / 'initial_temperature.npy')

        original_conv = self.conv(ig, h, original_genam)
        rotated_original_conv = self.rotate_rank2(
            rotation_matrix, original_conv)
        rotated_conv = self.conv(ig, h, rotated_genam)
        self.print_vec(rotated_original_conv, 'rotation x IsoGCN', 10)
        self.print_vec(rotated_conv, 'IsoGCN x rotation', 10)
        self.print_vec(rotated_conv - rotated_original_conv, 'Diff', 10)
        np.testing.assert_array_almost_equal(
            rotated_conv, rotated_original_conv)

        # Use model directly with sparse
        torch_original_genam = self.load_genam(original_path, mode='torch')
        torch_rotated_genam = self.load_genam(rotated_path, mode='torch')

        torch_original_conv = ig(
            torch.from_numpy(h), torch_original_genam).numpy()
        torch_rotated_original_conv = self.rotate_rank2(
            rotation_matrix, torch_original_conv)
        torch_rotated_conv = ig(
            torch.from_numpy(h), torch_rotated_genam).numpy()
        self.print_vec(torch_rotated_original_conv, 'rotation x IsoGCN', 10)
        self.print_vec(torch_rotated_conv, 'IsoGCN x rotation', 10)
        self.print_vec(
            torch_rotated_conv - torch_rotated_original_conv, 'Diff', 10)
        np.testing.assert_array_almost_equal(
            torch_rotated_conv, torch_rotated_original_conv)

        # Use network
        net_original_genam = self.load_genam(original_path, mode='sparse_info')
        net_rotated_genam = self.load_genam(rotated_path, mode='sparse_info')
        net_original_conv = tr.model(
            {'x': torch.from_numpy(h), 'supports': net_original_genam}
        ).detach().numpy()
        net_rotated_original_conv = self.rotate_rank2(
            rotation_matrix, net_original_conv)
        net_rotated_conv = tr.model(
            {'x': torch.from_numpy(h), 'supports': net_rotated_genam}
        ).detach().numpy()
        self.print_vec(net_rotated_original_conv, 'rotation x IsoGCN', 10)
        self.print_vec(net_rotated_conv, 'IsoGCN x rotation', 10)
        self.print_vec(
            net_rotated_conv - net_rotated_original_conv, 'Diff', 10)
        np.testing.assert_array_almost_equal(
            net_rotated_conv, net_rotated_original_conv)

        ir = inferer.Inferer(main_setting)
        inference_outpout_directory = \
            main_setting.trainer.output_directory / 'inferred'
        if inference_outpout_directory.exists():
            shutil.rmtree(inference_outpout_directory)
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            preprocessed_data_directory=[original_path, rotated_path],
            converter_parameters_pkl=Path(
                'tests/data/rotation_thermal_stress/preprocessed/'
                'preprocessors.pkl'),
            output_directory=inference_outpout_directory,
            overwrite=True, perform_postprocess=False)

        inferred_rotated_conv = results[1]['dict_y']['nodal_strain_mat']
        self.print_vec(net_rotated_conv, 'net IsoGCN x rotation', 5)
        self.print_vec(inferred_rotated_conv, 'inferred IsoGCN x rotation', 5)
        self.print_vec(
            inferred_rotated_conv - net_rotated_conv, 'Diff', 5)
        np.testing.assert_array_almost_equal(
            inferred_rotated_conv, net_rotated_conv)

        inferred_original_conv = results[0]['dict_y']['nodal_strain_mat']
        np.testing.assert_array_almost_equal(
            inferred_original_conv, net_original_conv)

        # Confirm answer has rotation equivariance
        rotated_original_answer = np.array([
            rotation_matrix @ r[..., 0] @ rotation_matrix.T for r
            in results[0]['dict_x']['nodal_strain_mat']])[..., None]
        answer = results[1]['dict_x']['nodal_strain_mat']
        self.print_vec(
            self.rotate_rank2(
                rotation_matrix, results[0]['dict_x']['nodal_strain_mat'])
            - answer, 'diff', 5)
        np.testing.assert_almost_equal(
            self.rotate_rank2(
                rotation_matrix, results[0]['dict_x']['nodal_strain_mat']),
            answer)
        np.testing.assert_almost_equal(rotated_original_answer, answer)

        # Confirm inference result has rotation equivariance
        rotated_original = np.array([
            rotation_matrix @ r[..., 0] @ rotation_matrix.T for r
            in results[0]['dict_y']['nodal_strain_mat']])[..., None]
        print(rotated_original[:3, :, :, 0])
        print(results[1]['dict_y']['nodal_strain_mat'][:3, :, :, 0])
        print(
            rotated_original[:3, :, :, 0]
            - results[1]['dict_y']['nodal_strain_mat'][:3, :, :, 0])
        np.testing.assert_almost_equal(
            rotated_original,
            results[1]['dict_y']['nodal_strain_mat'], decimal=5)

    def test_iso_gcn_rotation_thermal_stress_rank0_rank2_with_postprocess(
            self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation_thermal_stress/iso_gcn_rank0_rank2.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        tr.train()

        original_path = Path(
            'tests/data/rotation_thermal_stress/preprocessed/cube/original')
        rotated_path = Path(
            'tests/data/rotation_thermal_stress/preprocessed/cube/rotated')
        rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        ir = inferer.Inferer(main_setting)
        inference_outpout_directory = \
            main_setting.trainer.output_directory / 'inferred'
        if inference_outpout_directory.exists():
            shutil.rmtree(inference_outpout_directory)
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            preprocessed_data_directory=[original_path, rotated_path],
            converter_parameters_pkl=Path(
                'tests/data/rotation_thermal_stress/preprocessed/'
                'preprocessors.pkl'),
            output_directory=inference_outpout_directory,
            overwrite=True, perform_postprocess=True)

        # Confirm inference result has rotation equivariance
        rotated_original = np.array([
            rotation_matrix @ r[..., 0] @ rotation_matrix.T for r
            in results[0]['dict_y']['nodal_strain_mat']])[..., None]
        print(rotated_original[:3, :, :, 0])
        print(results[1]['dict_y']['nodal_strain_mat'][:3, :, :, 0])
        print(
            rotated_original[:3, :, :, 0]
            - results[1]['dict_y']['nodal_strain_mat'][:3, :, :, 0])
        np.testing.assert_almost_equal(
            rotated_original,
            results[1]['dict_y']['nodal_strain_mat'], decimal=5)

    def test_iso_gcn_rotation_thermal_stress_rank2_rank2(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation_thermal_stress/iso_gcn_rank2_rank2.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        tr.train()

        ir = inferer.Inferer(main_setting)
        inference_outpout_directory = \
            main_setting.trainer.output_directory / 'inferred'
        if inference_outpout_directory.exists():
            shutil.rmtree(inference_outpout_directory)
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            preprocessed_data_directory=[
                Path(
                    'tests/data/rotation_thermal_stress/preprocessed/cube/'
                    'original'),
                Path(
                    'tests/data/rotation_thermal_stress/preprocessed/cube/'
                    'rotated')],
            converter_parameters_pkl=Path(
                'tests/data/rotation_thermal_stress/preprocessed/'
                'preprocessors.pkl'),
            output_directory=inference_outpout_directory,
            overwrite=True)

        rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        # Confirm answer has rotation equivariance
        rotated_original_answer = np.array([
            rotation_matrix @ r[..., 0] @ rotation_matrix.T for r
            in results[0]['dict_x']['nodal_lte_mat']])[..., None]
        answer = results[1]['dict_x']['nodal_lte_mat']
        np.testing.assert_almost_equal(rotated_original_answer, answer)

        # Confirm inference result has rotation equivariance
        rotated_original = np.array([
            rotation_matrix @ r[..., 0] @ rotation_matrix.T for r
            in results[0]['dict_y']['nodal_lte_mat']])[..., None]
        print(rotated_original[:3, :, :, 0])
        print(results[1]['dict_y']['nodal_lte_mat'][:3, :, :, 0])
        print(
            rotated_original[:3, :, :, 0]
            - results[1]['dict_y']['nodal_lte_mat'][:3, :, :, 0])
        np.testing.assert_almost_equal(
            rotated_original,
            results[1]['dict_y']['nodal_lte_mat'], decimal=4)

    def test_iso_gcn_rotation_thermal_stress_dict_input(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation_thermal_stress/iso_gcn_dict_input.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        loss = tr.train()
        np.testing.assert_array_less(loss, 5.)

        ir = inferer.Inferer(main_setting)
        inference_outpout_directory = \
            main_setting.trainer.output_directory / 'inferred'
        if inference_outpout_directory.exists():
            shutil.rmtree(inference_outpout_directory)
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            preprocessed_data_directory=[
                Path(
                    'tests/data/rotation_thermal_stress/preprocessed/cube/'
                    'original'),
                Path(
                    'tests/data/rotation_thermal_stress/preprocessed/cube/'
                    'rotated')],
            converter_parameters_pkl=Path(
                'tests/data/rotation_thermal_stress/preprocessed/'
                'preprocessors.pkl'),
            output_directory=inference_outpout_directory,
            overwrite=True)

        rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        # Confirm answer has rotation equivariance
        rotated_original_answer = np.array([
            rotation_matrix @ r[..., 0] @ rotation_matrix.T for r
            in results[0]['dict_x']['nodal_strain_mat']])[..., None]
        answer = results[1]['dict_x']['nodal_strain_mat']
        np.testing.assert_almost_equal(rotated_original_answer, answer)

        # Confirm inference result has rotation equivariance
        rotated_original = np.array([
            rotation_matrix @ r[..., 0] @ rotation_matrix.T for r
            in results[0]['dict_y']['nodal_strain_mat']])[..., None]
        print(rotated_original[:3, :, :, 0])
        print(results[1]['dict_y']['nodal_strain_mat'][:3, :, :, 0])
        print(
            rotated_original[:3, :, :, 0]
            - results[1]['dict_y']['nodal_strain_mat'][:3, :, :, 0])
        np.testing.assert_almost_equal(
            rotated_original,
            results[1]['dict_y']['nodal_strain_mat'], decimal=5)

    def test_iso_gcn_rotation_thermal_stress_dict_input_dict_output(self):
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress'
            '/iso_gcn_dict_input_dict_output.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        tr.train()

        ir = inferer.Inferer(main_setting)
        inference_outpout_directory = \
            main_setting.trainer.output_directory / 'inferred'
        if inference_outpout_directory.exists():
            shutil.rmtree(inference_outpout_directory)
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            preprocessed_data_directory=[
                Path(
                    'tests/data/rotation_thermal_stress/preprocessed/cube/'
                    'original'),
                Path(
                    'tests/data/rotation_thermal_stress/preprocessed/cube/'
                    'rotated')],
            converter_parameters_pkl=Path(
                'tests/data/rotation_thermal_stress/preprocessed/'
                'preprocessors.pkl'),
            output_directory=inference_outpout_directory,
            overwrite=True, perform_postprocess=False)

        rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        # Confirm input has rotation equivariance
        rotated_original_input_data = np.array([
            rotation_matrix @ r[..., 0] @ rotation_matrix.T for r
            in results[0]['dict_x']['nodal_strain_mat']])[..., None]
        input_data = results[1]['dict_x']['nodal_strain_mat']
        np.testing.assert_almost_equal(rotated_original_input_data, input_data)

        name_answer = 'global_lte_mat'

        # Confirm answer has rotation equivariance
        rotated_original_answer = np.array([
            rotation_matrix @ r[..., 0] @ rotation_matrix.T for r
            in results[0]['dict_x'][name_answer]])[..., None]
        answer = results[1]['dict_x'][name_answer]
        np.testing.assert_almost_equal(rotated_original_answer, answer)

        # Confirm inference result has rotation invariance and equivariance
        rotated_original = self.rotate_rank2(
            rotation_matrix, results[0]['dict_y'][name_answer])
        print(rotated_original[0, :, :, 0])
        print(results[1]['dict_y'][name_answer][0, :, :, 0])
        print(
            rotated_original[0, :, :, 0]
            - results[1]['dict_y'][name_answer][0, :, :, 0])
        np.testing.assert_almost_equal(
            rotated_original,
            results[1]['dict_y'][name_answer], decimal=5)
        print(results[0]['dict_y']['cnt_temperature'][:10])
        print(results[1]['dict_y']['cnt_temperature'][:10])
        print(
            results[0]['dict_y']['cnt_temperature'][:10]
            - results[1]['dict_y']['cnt_temperature'][:10])
        np.testing.assert_almost_equal(
            results[0]['dict_y']['cnt_temperature'],
            results[1]['dict_y']['cnt_temperature'], decimal=5)

    def test_iso_gcn_rotation_thermal_stress_rank0_rank2_global_pooling(self):
        main_setting = setting.MainSetting.read_settings_yaml(Path(
            'tests/data/rotation_thermal_stress/iso_gcn_rank0_rank2_pool.yml'))
        tr = trainer.Trainer(main_setting)
        if tr.setting.trainer.output_directory.exists():
            shutil.rmtree(tr.setting.trainer.output_directory)
        tr.train()

        original_path = Path(
            'tests/data/rotation_thermal_stress/preprocessed/cube/original')
        rotated_path = Path(
            'tests/data/rotation_thermal_stress/preprocessed/cube/rotated')
        rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        ir = inferer.Inferer(main_setting)
        inference_outpout_directory = \
            main_setting.trainer.output_directory / 'inferred'
        if inference_outpout_directory.exists():
            shutil.rmtree(inference_outpout_directory)
        results = ir.infer(
            model=main_setting.trainer.output_directory,
            preprocessed_data_directory=[original_path, rotated_path],
            converter_parameters_pkl=Path(
                'tests/data/rotation_thermal_stress/preprocessed/'
                'preprocessors.pkl'),
            output_directory=inference_outpout_directory,
            overwrite=True, perform_postprocess=False)

        # Confirm answer has rotation equivariance
        rotated_original_answer = np.array([
            rotation_matrix @ r[..., 0] @ rotation_matrix.T for r
            in results[0]['dict_x']['global_lte_mat']])[..., None]
        answer = results[1]['dict_x']['global_lte_mat']
        self.print_vec(
            self.rotate_rank2(
                rotation_matrix, results[0]['dict_x']['global_lte_mat'])
            - answer, 'diff', 5)
        np.testing.assert_almost_equal(
            self.rotate_rank2(
                rotation_matrix, results[0]['dict_x']['global_lte_mat']),
            answer)
        np.testing.assert_almost_equal(rotated_original_answer, answer)

        # Confirm inference result has rotation equivariance
        rotated_original = np.array([
            rotation_matrix @ r[..., 0] @ rotation_matrix.T for r
            in results[0]['dict_y']['global_lte_mat']])[..., None]
        print(rotated_original[:3, :, :, 0])
        print(results[1]['dict_y']['global_lte_mat'][:, :, :, 0])
        print(
            rotated_original[:3, :, :, 0]
            - results[1]['dict_y']['global_lte_mat'][:, :, :, 0])
        np.testing.assert_almost_equal(
            rotated_original,
            results[1]['dict_y']['global_lte_mat'], decimal=5)

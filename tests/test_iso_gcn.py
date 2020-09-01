
from pathlib import Path
import shutil
import unittest

import numpy as np
import torch

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

    def generate_rotation_matrix(self):
        def normalize(x):
            return x / np.linalg.norm(x)

        vec1 = normalize(np.random.rand(3)*2 - 1)
        vec2 = normalize(np.random.rand(3)*2 - 1)
        vec3 = normalize(np.cross(vec1, vec2))
        vec2 = np.cross(vec3, vec1)
        return np.array([vec1, vec2, vec3])

    def print_vec(self, x, name=None):
        print('--')
        if name is not None:
            print(name)
        if len(x.shape) == 4:
            for _x in x[..., 0]:
                print(_x)
        else:
            for _x in x:
                print(_x[:, 0])
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
            results[1]['dict_x']['initial_temperature'], decimal=3)
        np.testing.assert_almost_equal(
            results[0]['dict_y']['initial_temperature'],
            results[1]['dict_y']['initial_temperature'], decimal=3)

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
            results[1]['dict_y']['nodal_lte_mat'], decimal=3)

    def test_iso_gcn_rotation_thermal_stress_dict_input(self):
        main_setting = setting.MainSetting.read_settings_yaml(
            Path('tests/data/rotation_thermal_stress/iso_gcn_thermal.yml'))
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
            results[1]['dict_y']['nodal_strain_mat'], decimal=3)

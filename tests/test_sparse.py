
from pathlib import Path
import unittest

import numpy as np
import scipy.sparse as sp
import torch

import siml.datasets as datasets
import siml.networks.sparse as sparse


PLOT = False


class TestPENN(unittest.TestCase):

    def test_linear_penn_convolution_same_as_isogcn(self):
        data_path = Path(
            'tests/data/heat_boundary/preprocessed/cylinder/clscale0.3/'
            'steepness1.0_rep0')
        inc_int = sp.load_npz(data_path / 'inc_int.npz')
        n = inc_int.shape[-1]
        phi = torch.rand(n, 3, 3, 10)

        supports = datasets.convert_sparse_tensor([
            datasets.pad_sparse(inc_int)])
        res = sparse.mul(supports[0], phi).detach().numpy()

        for i in range(3):
            for j in range(3):
                np.testing.assert_almost_equal(
                    res[:, i, j, :], inc_int.dot(phi[:, i, j, :]))

import pathlib

import numpy as np
import pytest
import scipy.sparse as sp
import torch

import siml
import siml.datasets as datasets
from siml.networks.implicit_gnn import ImplicitGNN


@pytest.fixture
def prepare_dataset():
    data_path = pathlib.Path('tests/data/grad/interim/train/0')
    nadj = sp.load_npz(data_path / 'nodal_nadj.npz')
    supports = datasets.convert_sparse_tensor([
        datasets.pad_sparse(nadj)
    ])

    np_phi = np.load(data_path / 'phi.npy').astype(np.float32)
    phi = torch.from_numpy(np_phi)
    return supports, phi


def test__run_implicit_gnn(prepare_dataset):
    supports, phi = prepare_dataset
    n_nodes = phi.shape[1]
    block_setting = siml.setting.BlockSetting(
        type="implicit_gnn",
        nodes=[n_nodes, n_nodes],
        support_input_index=0,
        activations=["tanh"]
    )

    model = ImplicitGNN(block_setting)
    h = model.forward(x=phi, supports=supports)

from siml import setting
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import pytest

import siml.datasets as datasets
from siml.networks.gcn import GCN


@pytest.fixture
def prepare_dataset():
    data_path = Path('tests/data/grad/interim/train/0')
    gx = sp.load_npz(data_path / 'nodal_grad_x.npz')
    supports = datasets.convert_sparse_tensor([
        datasets.pad_sparse(gx)])

    np_phi = np.load(data_path / 'phi.npy').astype(np.float32)
    phi = torch.from_numpy(np_phi)
    return supports, phi


def test__run_gcn_no_weight(prepare_dataset):
    supports, phi = prepare_dataset
    gcn_no_weight = GCN(
        block_setting=setting.BlockSetting(
            type='gcn',
            support_input_index=0,
            optional={
                "create_subchains": False
            }

        )
    )
    gcn_no_weight.forward(
        x=phi,
        supports=supports
    )


def test__gcn_no_weight_has_no_parameters():
    gcn_no_weight = GCN(
        block_setting=setting.BlockSetting(
            type='gcn',
            support_input_index=0,
            optional={
                "create_subchains": False
            }

        )
    )

    assert len(list(gcn_no_weight.parameters())) == 0


def test__compare_std_gcn_no_weight(prepare_dataset):
    supports, phi = prepare_dataset
    gcn_no_weight = GCN(
        block_setting=setting.BlockSetting(
            type='gcn',
            support_input_index=0,
            optional={
                "create_subchains": False
            }

        )
    )
    h = gcn_no_weight.forward(
        x=phi,
        supports=supports
    )

    std_original = torch.std(phi)
    std_update = torch.std(h)

    # No weight GCN has a role to smooth scalar field
    # Thus, std_original is larger than std_update
    assert std_original > std_update

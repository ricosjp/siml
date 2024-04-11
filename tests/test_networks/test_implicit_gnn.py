import pathlib
import shutil

import numpy as np
import pytest
import scipy.sparse as sp
import torch

import siml
import siml.datasets as datasets
from siml.networks.implicit_gnn import ImplicitGNN, ImplicitFunction


def test__train_implicit_gnn():
    main_setting = siml.setting.MainSetting.read_settings_yaml(
        pathlib.Path("tests/data/grad/implicit_gnn.yml")
    )

    trainer = siml.trainer.Trainer(main_setting)
    if trainer.setting.trainer.output_directory.exists():
        shutil.rmtree(trainer.setting.trainer.output_directory)

    loss = trainer.train()
    np.testing.assert_array_less(loss, .1)


@pytest.mark.parametrize("settings", [
    {"activations": ["tanh", "tanh"], "nodes": [10, 10, 10]},
    {"activations": [], "nodes": [10]},
])
def test__not_accepting_settings(settings: dict):

    block_settings = siml.setting.BlockSetting(
        **settings
    )
    with pytest.raises(ValueError):
        _ = ImplicitGNN(block_settings)


# region test fot ImplicitFunctions

def create_linked_adjacency_matrix(n_point: int) -> sp.coo_matrix:
    adj = np.zeros((n_point, n_point), dtype=np.float32)

    for i in range(n_point):
        if i == 0:
            adj[0][1] = 1
            continue
        if i == n_point - 1:
            adj[i][i - 1] = 1
            continue
        
        adj[i][i - 1] = 1
        adj[i][i + 1] = 1


    adj = sp.coo_matrix(adj).tocsr()
    return adj


def test__sample_equiblium_equation():
    n_point = 10
    adj = create_linked_adjacency_matrix(n_point)
    adj *= 0.5  # in order for I - A to have inverse matrix
    adj = torch.sparse_coo_tensor(adj.nonzero(), adj.data, adj.shape)

    X_init = torch.tensor(np.random.rand(1, n_point), dtype=torch.float32)
    B = torch.tensor(
        np.array(list(range(n_point))).reshape(1, n_point),
        dtype=torch.float32
    )
    W = torch.eye(1, dtype=torch.float32)

    X_new, _, status, _ = ImplicitFunction._forward_iteration(
        W=W,
        X=X_init,
        A=adj,
        B=B,
        phi=lambda x: x
    )

    # X = W X A + B
    # When W = Identity
    # X = B (I - A)^-1
    inv = torch.linalg.inv(torch.eye(n_point) - adj)
    desired = B @ inv

    np.testing.assert_array_almost_equal(
        X_new.detach().numpy(), desired.detach().numpy(), decimal=1e-3
    )

# endregion

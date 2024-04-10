import pathlib
import shutil

import numpy as np
import pytest
import scipy.sparse as sp
import torch

import siml
import siml.datasets as datasets
from siml.networks.implicit_gnn import ImplicitGNN


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
def test__not_accepting_settings(settings):

    block_settings = siml.setting.BlockSetting(
        **settings
    )
    with pytest.raises(ValueError):
        _ = ImplicitGNN(block_settings)

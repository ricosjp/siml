import pathlib
import secrets

import pytest

from siml.path_like_objects import SimlFileBulider
from siml.path_like_objects.siml_files import (
    ISimlCheckpointFile,
    ISimlNumpyFile, ISimlPickleFile,
    ISimlYamlFile
)

TEST_ENCRYPT_KEY = secrets.token_bytes(32)


@pytest.mark.parametrize("path", [
    "./sample.yml", "./sample.yml.enc"
])
def test__create_yaml_file(path):
    path = pathlib.Path(path)
    siml_path = SimlFileBulider.yaml_file(path)

    assert isinstance(siml_path, ISimlYamlFile)


@pytest.mark.parametrize("path", [
    "./sample.npy", "./sample.npy.enc",
    "./sample.npz", "./sample.npz.enc"
])
def test__create_npy_file(path):
    path = pathlib.Path(path)
    siml_path = SimlFileBulider.numpy_file(path)

    assert isinstance(siml_path, ISimlNumpyFile)


@pytest.mark.parametrize("path", [
    "./sample.pth", "./sample.pth.enc"
])
def test__create_pth_file(path):
    path = pathlib.Path(path)
    siml_path = SimlFileBulider.checkpoint_file(path)

    assert isinstance(siml_path, ISimlCheckpointFile)


@pytest.mark.parametrize("path", [
    "./sample.pkl", "./sample.pkl.enc"
])
def test__create_pickle_file(path):
    path = pathlib.Path(path)
    siml_path = SimlFileBulider.pickle_file(path)

    assert isinstance(siml_path, ISimlPickleFile)

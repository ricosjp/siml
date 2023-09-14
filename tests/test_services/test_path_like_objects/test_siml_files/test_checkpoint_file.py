import pathlib
import shutil
import secrets

import numpy as np
import pytest
import torch

from siml.path_like_objects.siml_files import SimlCheckpointFile


TEST_ENCRYPT_KEY = secrets.token_bytes(32)


@pytest.mark.parametrize("path, ext", [
    ("./sample/sample.pth", ".pth"),
    ("./sample/sample.pth.enc", ".pth.enc")
])
def test__check_extension_type(path, ext):
    path = pathlib.Path(path)
    siml_path = SimlCheckpointFile(path)
    assert siml_path._ext_type.value == ext
    assert siml_path.file_path == path


@pytest.mark.parametrize("path", [
    ("./sample/sample.npy"),
    ("./sample/sample.npy.enc")
])
def test__check_error_extension_type(path):
    path = pathlib.Path(path)
    with pytest.raises(NotImplementedError):
        _ = SimlCheckpointFile(path)


@pytest.mark.parametrize("path, enc", [
    ("./sample/sample.pth", False),
    ("./sample/sample.pth.enc", True)
])
def test__is_encrypted(path, enc):
    path = pathlib.Path(path)
    siml_path = SimlCheckpointFile(path)
    assert siml_path.is_encrypted == enc


@pytest.fixture(scope="module")
def create_test_dir():
    directory = pathlib.Path(__file__).parent
    test_dir = (directory / "data")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()


def test__save_and_load(create_test_dir):
    sample_array = np.random.rand(3, 4)
    sample_tensor = torch.tensor(sample_array)

    path = pathlib.Path(__file__).parent / "data/sample.pth"
    siml_path = SimlCheckpointFile(path)
    siml_path.save(sample_tensor, overwrite=True)

    assert path.exists()

    loaded_tensor: torch.Tensor = siml_path.load(device="cpu")
    loaded_array = loaded_tensor.detach().numpy()
    np.testing.assert_array_almost_equal(
        loaded_array, sample_array
    )


def test__save_encrypted_and_load(create_test_dir):
    sample_array = np.random.rand(3, 4)
    sample_tensor = torch.tensor(sample_array)

    path = pathlib.Path(__file__).parent / "data/sample_2.pth.enc"
    siml_path = SimlCheckpointFile(path)
    siml_path.save(
        sample_tensor,
        overwrite=True,
        encrypt_key=TEST_ENCRYPT_KEY
    )

    assert path.exists()

    loaded_tensor: torch.Tensor = siml_path.load(
        device="cpu", decrypt_key=TEST_ENCRYPT_KEY
    )
    loaded_array = loaded_tensor.detach().numpy()
    np.testing.assert_array_almost_equal(
        loaded_array, sample_array
    )


def test__save_not_allowed_overwrite(create_test_dir):
    sample_array = np.random.rand(3, 4)
    sample_tensor = torch.tensor(sample_array)

    path = pathlib.Path(__file__).parent / "data/sample.pth"
    path.touch(exist_ok=True)

    siml_path = SimlCheckpointFile(path)
    with pytest.raises(FileExistsError):
        siml_path.save(sample_tensor, overwrite=False)


@pytest.mark.parametrize("path, num_epoch", [
    ("./aaa/snapshot_epoch_1.pth", 1),
    ("./aaa/snapshot_epoch_123.pth.enc", 123),
    ("./aaa/snapshot_epoch_30.pth.enc", 30)
])
def test__get_epoch(path, num_epoch):
    siml_path = SimlCheckpointFile(pathlib.Path(path))

    assert siml_path.epoch == num_epoch


@pytest.mark.parametrize("path", [
    ("./aaa/deployed_1.pth"),
    ("./aaa/model.pth"),
    ("./aaa/epoch_2.pth")
])
def test__get_epoch_not_handled(path):
    siml_path = SimlCheckpointFile(pathlib.Path(path))

    with pytest.raises(ValueError):
        _ = siml_path.epoch

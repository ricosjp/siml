import pathlib
import secrets
import shutil

import pytest
import torch

from siml.path_like_objects.siml_files import SimlYamlFile

TEST_ENCRYPT_KEY = secrets.token_bytes(32)


@pytest.mark.parametrize("path, ext", [
    ("./sample/sample.yml", ".yml"),
    ("./sample/sample.yml.enc", ".yml.enc")
])
def test__check_extension_type(path, ext):
    path = pathlib.Path(path)
    siml_path = SimlYamlFile(path)
    assert siml_path._ext_type.value == ext
    assert siml_path.file_path == path


@pytest.mark.parametrize("path", [
    ("./sample/sample.npy"),
    ("./sample/sample.npy.enc")
])
def test__check_error_extension_type(path):
    path = pathlib.Path(path)
    with pytest.raises(NotImplementedError):
        _ = SimlYamlFile(path)


@pytest.mark.parametrize("path, enc", [
    ("./sample/sample.yml", False),
    ("./sample/sample.yml.enc", True)
])
def test__is_encrypted(path, enc):
    path = pathlib.Path(path)
    siml_path = SimlYamlFile(path)
    assert siml_path.is_encrypted == enc


@pytest.fixture(scope="module")
def create_test_dir():
    directory = pathlib.Path(__file__).parent
    test_dir = (directory / "data")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()


def test__save_and_load(create_test_dir):
    sample_data = {"a": 1, "b": 2}

    path = pathlib.Path(__file__).parent / "data/sample.yml"
    siml_path = SimlYamlFile(path)
    siml_path.save(sample_data, overwrite=True)

    assert path.exists()

    loaded_data: torch.Tensor = siml_path.load()
    assert loaded_data == sample_data


def test__save_encrypted_and_load(create_test_dir):
    sample_data = {"a": 1, "b": 2}

    path = pathlib.Path(__file__).parent / "data/sample_2.yml.enc"
    siml_path = SimlYamlFile(path)
    siml_path.save(
        sample_data,
        overwrite=True,
        encrypt_key=TEST_ENCRYPT_KEY
    )

    assert path.exists()

    loaded_data = siml_path.load(decrypt_key=TEST_ENCRYPT_KEY)
    assert loaded_data == sample_data


def test__save_not_allowed_overwrite(create_test_dir):
    sample_data = {"a": 1, "b": 2}

    path = pathlib.Path(__file__).parent / "data/sample.yml"
    path.touch(exist_ok=True)

    siml_path = SimlYamlFile(path)
    with pytest.raises(FileExistsError):
        siml_path.save(sample_data, overwrite=False)

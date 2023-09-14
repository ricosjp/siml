import pathlib
import secrets
import shutil

import pytest

from siml.path_like_objects.siml_files import SimlPickleFile


TEST_ENCRYPT_KEY = secrets.token_bytes(32)


@pytest.mark.parametrize("path, ext", [
    ("./sample/sample.pkl", ".pkl"),
    ("./sample/sample.pkl.enc", ".pkl.enc")
])
def test__check_extension_type(path, ext):
    path = pathlib.Path(path)
    siml_path = SimlPickleFile(path)
    assert siml_path._ext_type.value == ext
    assert siml_path.file_path == path


@pytest.mark.parametrize("path", [
    ("./sample/sample.npy"),
    ("./sample/sample.npy.enc")
])
def test__check_error_extension_type(path):
    path = pathlib.Path(path)
    with pytest.raises(NotImplementedError):
        _ = SimlPickleFile(path)


@pytest.mark.parametrize("path, enc", [
    ("./sample/sample.pkl", False),
    ("./sample/sample.pkl.enc", True)
])
def test__is_encrypted(path, enc):
    path = pathlib.Path(path)
    siml_path = SimlPickleFile(path)
    assert siml_path.is_encrypted == enc


@pytest.fixture(scope="module")
def create_test_dir():
    directory = pathlib.Path(__file__).parent
    test_dir = (directory / "data")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()


def test__save_and_load(create_test_dir):
    sample_dict = {"a": 1, "b": 2}

    path = pathlib.Path(__file__).parent / "data/sample.pkl"
    siml_path = SimlPickleFile(path)
    siml_path.save(sample_dict, overwrite=True)

    assert path.exists()

    loaded = siml_path.load()
    assert loaded == sample_dict


def test__save_encrypted_and_load(create_test_dir):
    sample_dict = {"a": 1, "b": 2}

    path = pathlib.Path(__file__).parent / "data/sample_2.pkl.enc"
    siml_path = SimlPickleFile(path)
    siml_path.save(
        sample_dict,
        encrypt_key=TEST_ENCRYPT_KEY,
        overwrite=True
    )

    assert path.exists()

    loaded = siml_path.load(decrypt_key=TEST_ENCRYPT_KEY)
    assert loaded == sample_dict


def test__save_not_allowed_overwrite(create_test_dir):
    sample_dict = {"a": 1, "b": 2}

    path = pathlib.Path(__file__).parent / "data/sample.pkl"
    path.touch(exist_ok=True)

    siml_path = SimlPickleFile(path)
    with pytest.raises(FileExistsError):
        siml_path.save(sample_dict)

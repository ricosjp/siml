import pathlib
import shutil
import secrets

import numpy as np
import scipy.sparse as sp
import pytest

from siml.path_like_objects.siml_files import SimlNumpyFile


TEST_ENCRYPT_KEY = secrets.token_bytes(32)


@pytest.mark.parametrize("path, ext", [
    ("./sample/sample.npy", ".npy"),
    ("./sample/sample.npy.enc", ".npy.enc"),
    ("./sample/sample.npz", ".npz"),
    ("./sample/sample.npz.enc", ".npz.enc"),
])
def test__check_extension_type(path, ext):
    path = pathlib.Path(path)
    siml_path = SimlNumpyFile(path)
    assert siml_path._ext_type.value == ext
    assert siml_path.file_path == path


@pytest.mark.parametrize("path", [
    ("./sample/sample.pkl"),
    ("./sample/sample.pkl.enc")
])
def test__check_error_extension_type(path):
    path = pathlib.Path(path)
    with pytest.raises(NotImplementedError):
        _ = SimlNumpyFile(path)


@pytest.mark.parametrize("path, enc", [
    ("./sample/sample.npy", False),
    ("./sample/sample.npz", False),
    ("./sample/sample.npy.enc", True),
    ("./sample/sample.npz.enc", True),
])
def test__is_encrypted(path, enc):
    path = pathlib.Path(path)
    siml_path = SimlNumpyFile(path)
    assert siml_path.is_encrypted == enc


@pytest.fixture(scope="module")
def create_test_dir():
    directory = pathlib.Path(__file__).parent
    test_dir = (directory / "data")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()


def test__save_npy_and_load(create_test_dir):
    sample_array = np.random.rand(3, 4)

    path = pathlib.Path(__file__).parent / "data/sample.npy"
    siml_path = SimlNumpyFile(path)
    siml_path.save(sample_array, overwrite=True)

    assert path.exists()

    loaded_array: np.ndarray = siml_path.load()
    np.testing.assert_array_almost_equal(
        loaded_array, sample_array
    )


def test__save_npz_and_load(create_test_dir):
    sample_array = sp.csr_matrix((3, 4), dtype=np.float32)

    path = pathlib.Path(__file__).parent / "data/sample.npz"
    siml_path = SimlNumpyFile(path)
    siml_path.save(sample_array, overwrite=True)

    assert path.exists()

    loaded_array: sp.csr_matrix = siml_path.load()
    np.testing.assert_array_almost_equal(
        loaded_array.todense(), sample_array.todense()
    )


def test__save_encrypted_and_load(create_test_dir):
    sample_array = np.random.rand(3, 4)

    path = pathlib.Path(__file__).parent / "data/sample.npy.enc"
    siml_path = SimlNumpyFile(path)
    siml_path.save(
        sample_array, overwrite=True, encrypt_key=TEST_ENCRYPT_KEY
    )

    assert path.exists()

    loaded_array: np.ndarray = siml_path.load(
        decrypt_key=TEST_ENCRYPT_KEY
    )
    np.testing.assert_array_almost_equal(
        loaded_array, sample_array
    )


def test__save_npz_encrypted_and_load(create_test_dir):
    sample_array = sp.csr_matrix((3, 4), dtype=np.float32)

    path = pathlib.Path(__file__).parent / "data/sample.npz.enc"
    siml_path = SimlNumpyFile(path)
    siml_path.save(
        sample_array, overwrite=True, encrypt_key=TEST_ENCRYPT_KEY
    )

    assert path.exists()

    loaded_array: sp.csr_matrix = siml_path.load(decrypt_key=TEST_ENCRYPT_KEY)
    np.testing.assert_array_almost_equal(
        loaded_array.todense(), sample_array.todense()
    )


def test__save_not_allowed_overwrite(create_test_dir):
    sample_array = np.random.rand(3, 4)

    path = pathlib.Path(__file__).parent / "data/sample.npy"
    path.touch(exist_ok=True)

    siml_path = SimlNumpyFile(path)
    with pytest.raises(FileExistsError):
        siml_path.save(sample_array, overwrite=False)

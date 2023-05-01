import pytest
import pathlib
import io
import shutil
import secrets
import pickle

import numpy as np
import scipy.sparse as sp

from siml import util
from siml.path_like_objects import SimlFileBulider


TEST_ENCRYPT_KEY = secrets.token_bytes(32)


@pytest.fixture(scope="module")
def create_test_tmp_data():
    directory = pathlib.Path(__file__).parent
    test_dir = (directory / "data")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    a = np.random.rand(3, 4)
    np.save(test_dir / "sample.npy", a)
    util.save_variable(
        test_dir,
        file_basename="sample",
        data=a,
        encrypt_key=TEST_ENCRYPT_KEY
    )

    b = sp.csr_matrix((3, 4), dtype=np.float32)
    util.save_variable(
        test_dir,
        file_basename="sample",
        data=b,
        encrypt_key=None
    )
    util.save_variable(
        test_dir,
        file_basename="sample",
        data=b,
        encrypt_key=TEST_ENCRYPT_KEY
    )

    c = [1, 2, 3]
    with open((test_dir / "sample.pkl"), 'wb') as fw:
        pickle.dump(c, fw)

    # Open the file to be pickled and read its content
    with open((test_dir / "sample.pkl"), 'rb') as f:
        data = f.read()

    util.encrypt_file(
        key=TEST_ENCRYPT_KEY,
        file_path=(test_dir / "sample.pkl.enc"),
        binary=io.BytesIO(data)
    )


@pytest.mark.usefixtures('create_test_tmp_data')
@pytest.mark.parametrize("rel_file_path", [
    ("data/sample.npy"),
    ("data/sample.npy.enc"),
    ("data/sample.npz"),
    ("data/sample.npz.enc"),
])
def test__initialize_numpy_file(rel_file_path):
    directory = pathlib.Path(__file__).parent
    file_path = (directory / rel_file_path)
    siml_file = SimlFileBulider.numpy_file(
        file_path
    )
    assert siml_file.file_path == file_path


@pytest.mark.usefixtures('create_test_tmp_data')
@pytest.mark.parametrize("rel_file_path", [
    ("data/sample.pkl"),
    ("data/sample.pkl.enc")
])
def test__initialize_pickle_file(rel_file_path):
    directory = pathlib.Path(__file__).parent
    file_path = (directory / rel_file_path)
    siml_file = SimlFileBulider.pickle_file(
        file_path
    )
    assert siml_file.file_path == file_path


@pytest.mark.usefixtures('create_test_tmp_data')
@pytest.mark.parametrize("rel_file_path", [
    ("data/sample.npy"),
    ("data/sample.npy.enc"),
    ("data/sample.npz"),
    ("data/sample.npz.enc"),
])
def test__load_numpy_content(rel_file_path):
    directory = pathlib.Path(__file__).parent
    file_path: pathlib.Path = (directory / rel_file_path)
    siml_file = SimlFileBulider.numpy_file(
        file_path
    )

    _ = siml_file.load(decrypt_key=TEST_ENCRYPT_KEY)


@pytest.mark.usefixtures('create_test_tmp_data')
@pytest.mark.parametrize("rel_file_path", [
    ("data/sample.pkl"),
    ("data/sample.pkl.enc")
])
def test__load_file_pickle_content(rel_file_path):
    directory = pathlib.Path(__file__).parent
    file_path: pathlib.Path = (directory / rel_file_path)
    siml_file = SimlFileBulider.pickle_file(
        file_path
    )

    _ = siml_file.load(decrypt_key=TEST_ENCRYPT_KEY)


@pytest.mark.usefixtures('create_test_tmp_data')
@pytest.mark.parametrize("not_enc_path, enc_path", [
    ("data/sample.npy", "data/sample.npy.enc"),
    ("data/sample.npz", "data/sample.npz.enc"),
])
def test__load_same_contents(not_enc_path, enc_path):
    directory = pathlib.Path(__file__).parent

    enc_siml_file = SimlFileBulider.numpy_file(
        (directory / enc_path)
    )
    siml_file = SimlFileBulider.numpy_file(
        (directory / not_enc_path)
    )

    data = siml_file.load()
    enc_data = enc_siml_file.load(decrypt_key=TEST_ENCRYPT_KEY)

    if isinstance(data, np.ndarray):
        np.testing.assert_almost_equal(data, enc_data)

    elif isinstance(data, sp.spmatrix):
        np.testing.assert_almost_equal(
            data.todense(),
            enc_data.todense()
        )
    else:
        pytest.fail()


@pytest.mark.usefixtures('create_test_tmp_data')
@pytest.mark.parametrize("not_enc_path, enc_path", [
    ("data/sample.pkl", "data/sample.pkl.enc")
])
def test__load_same_pickle_contents(not_enc_path, enc_path):
    directory = pathlib.Path(__file__).parent

    enc_siml_file = SimlFileBulider.pickle_file(
        (directory / enc_path)
    )
    siml_file = SimlFileBulider.pickle_file(
        (directory / not_enc_path)
    )

    data = siml_file.load()
    enc_data = enc_siml_file.load(decrypt_key=TEST_ENCRYPT_KEY)

    assert data == enc_data

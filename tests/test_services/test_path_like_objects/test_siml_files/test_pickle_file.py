import pathlib

import pytest

from siml.path_like_objects.siml_files import SimlPickleFile


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

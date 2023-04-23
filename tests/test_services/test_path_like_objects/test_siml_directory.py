import pathlib
import shutil

import pytest

from siml.path_like_objects import SimlDirectory

TEST_DATA_DIR = pathlib.Path(__file__).parent / "data_dir"


@pytest.fixture(scope="module")
def create_test_cases():
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)
    TEST_DATA_DIR.mkdir()

    (TEST_DATA_DIR / "variable1.npy").touch()
    (TEST_DATA_DIR / "variable2.npz.enc").touch()


@pytest.mark.parametrize("path", [
    "tests/data/deform",
    "tests/data/csv_prepost/raw"
])
def test__initialize(path):
    siml_dir = SimlDirectory(
        pathlib.Path(path)
    )

    assert siml_dir.path == pathlib.Path(path)


@pytest.mark.parametrize("variable_name, ext", [
    ("variable1", ".npy"),
    ("variable2", ".npz.enc")
])
def test__find_variable_file(variable_name, ext, create_test_cases):

    siml_dir = SimlDirectory(TEST_DATA_DIR)

    siml_file = siml_dir.find_variable_file(variable_name)
    assert siml_file.file_path == TEST_DATA_DIR / f"{variable_name}{ext}"
    assert siml_dir.exist_variable_file(variable_name)


@pytest.mark.parametrize("variable_name, ext", [
    ("variable3", ".npz"),
    ("variable4", ".npz.enc")
])
def test__failed_find_variable_file(variable_name, ext, create_test_cases):

    siml_dir = SimlDirectory(TEST_DATA_DIR)

    assert siml_dir.exist_variable_file(variable_name) is False
    with pytest.raises(ValueError):
        _ = siml_dir.find_variable_file(variable_name)


@pytest.mark.parametrize("variable_name, ext", [
    ("variable3", ".npz"),
    ("variable4", ".npz.enc")
])
def test__find_variable_file_as_None(variable_name, ext, create_test_cases):

    siml_dir = SimlDirectory(TEST_DATA_DIR)

    siml_file = siml_dir.find_variable_file(variable_name, allow_missing=True)
    assert siml_file is None
    assert siml_dir.exist_variable_file(variable_name) is False

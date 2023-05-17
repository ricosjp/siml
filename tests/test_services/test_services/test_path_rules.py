import pathlib
import shutil
import os

import pytest

from siml.services.path_rules import SimlPathRules
from siml.base.siml_enums import DirectoryType


@pytest.mark.parametrize("data_path, expect", [
    (pathlib.Path('data/c/raw/a/b'), DirectoryType.RAW),
    (pathlib.Path('data/interim'), DirectoryType.INTERIM),
    (pathlib.Path('data/aa/interim/cccc/ddd'), DirectoryType.INTERIM),
    (pathlib.Path('data/preprocessed/a/c'), DirectoryType.PREPROCESSED)
])
def test__detect_directory_type(data_path, expect):
    rule = SimlPathRules()
    assert rule.detect_directory_type(data_path) == expect
    assert rule.is_target_directory_type(data_path, expect)


@pytest.mark.parametrize("data_path", [
    (pathlib.Path('data/a/b')),
    (pathlib.Path('data/interims')),
    (pathlib.Path('data/')),
    (pathlib.Path('/'))
])
def test__none_directory_type(data_path):
    rule = SimlPathRules()
    assert rule.detect_directory_type(data_path) is None


@pytest.mark.parametrize("data_path, output_base, expect", [
    ("./aaa/bbb/interim/ddd", "./aaa/ccc", "./aaa/ccc/bbb/ddd"),
    ("data/a/raw/b/c", "data/output", "data/output/a/b/c"),
    ("data/preprocessed/b/c", "data/output/", "data/output/b/c"),
    ("data/a/b/c", "data/output/", "data/output")
])
def test__determine_output_directory(data_path, output_base, expect):
    rule = SimlPathRules()
    result = rule.determine_output_directory(
        pathlib.Path(data_path),
        pathlib.Path(output_base)
    )
    assert result == pathlib.Path(expect)


@pytest.mark.parametrize("input_dir, output_dir, str_replace, expect", [
    (
        pathlib.Path('data/raw/a/b'),
        pathlib.Path('data/sth'),
        'raw',
        pathlib.Path('data/sth/a/b')
    ),
    (
        pathlib.Path('tests/data/list/data/tet2_3_modulusx0.9000/interim'),
        pathlib.Path('tests/data/list/preprocessed'),
        'interim',
        pathlib.Path('tests/data/list/preprocessed/data/tet2_3_modulusx0.9000')
    )
])
def test__base_determine_output_directory(
        input_dir, output_dir, str_replace, expect):

    rules = SimlPathRules()
    assert rules._determine_output_directory(
        input_dir,
        output_dir,
        str_replace
    ) == expect


@pytest.mark.parametrize("data_directory, expect", [
    ("./sample/interim/aaa/bbb", None),
    ("./sample/raw/aaa/bbb", None),
])
def test__determine_write_simulation_case_dir_none_preprocessed(
    data_directory, expect
):
    rules = SimlPathRules()
    if expect is not None:
        pathlib.Path(expect).mkdir(exist_ok=True)

    actual = rules.determine_write_simulation_case_dir(
        data_directory=pathlib.Path(data_directory),
        write_simulation_base=None
    )
    assert actual is None


@pytest.fixture
def create_test_data_dir():
    test_dir = pathlib.Path(__file__).parent / "tmp_data_test_rules"
    if test_dir.exists():
        shutil.rmtree(test_dir)

    test_dir.mkdir()
    yield test_dir
    # teardown
    shutil.rmtree(test_dir)


@pytest.mark.parametrize("data_directory, expect", [
    ("preprocessed/aaa/bbb", "raw/aaa/bbb"),
    ("preprocessed/ccc/ddd", "interim/ccc/ddd"),
])
def test__determine_write_simulation_case_dir_none_base(
    data_directory, expect, create_test_data_dir
):
    test_dir: pathlib.Path = create_test_data_dir
    data_directory = test_dir / data_directory
    expect = test_dir / expect

    if expect is not None:
        pathlib.Path(expect).mkdir(exist_ok=True, parents=True)

    rules = SimlPathRules()
    actual = rules.determine_write_simulation_case_dir(
        data_directory=data_directory,
        write_simulation_base=None
    )
    assert actual == expect


@pytest.mark.parametrize("data_directory, write_base, expect", [
    ("./sample/raw/aaa/bbb", "./any", "./sample/raw/aaa/bbb"),
    ("./sample/raw/aaa/bbb", "./sample", "./sample/raw/aaa/bbb"),
    ("./sample/interim/aaa/bbb", "./sample", "./sample/aaa/bbb"),
    ("./sample/interim/aaa/bbb", "./raw", "./raw/sample/aaa/bbb"),
])
def test__determine_write_simulation_case_dir(
    data_directory, write_base, expect
):
    data_directory = pathlib.Path(data_directory)
    expect = pathlib.Path(expect)

    if expect is not None:
        pathlib.Path(expect).mkdir(exist_ok=True, parents=True)

    rules = SimlPathRules()
    actual = rules.determine_write_simulation_case_dir(
        data_directory=data_directory,
        write_simulation_base=pathlib.Path(write_base)
    )
    assert actual == expect


@pytest.mark.parametrize("directory, dir_type_to, expect", [
    ("./sample/aaa/bbb", DirectoryType.RAW, None),
    (
        "./sample/raw/aaa/bbb",
        DirectoryType.RAW,
        "./sample/raw/aaa/bbb"
    ),
    (
        "./sample/raw/aaa/bbb",
        DirectoryType.INTERIM,
        "./sample/interim/aaa/bbb"
    ),
    (
        "./sample/raw/aaa/bbb",
        DirectoryType.PREPROCESSED,
        "./sample/preprocessed/aaa/bbb"
    )
])
def test__switch_directory_type(directory, dir_type_to, expect):
    rules = SimlPathRules()
    actual = rules.switch_directory_type(
        pathlib.Path(directory),
        dir_type_to=dir_type_to
    )
    if expect is None:
        assert actual is None
    else:
        assert actual == pathlib.Path(expect)


@pytest.mark.parametrize("input_dir, output_dir, expect", [
    (
        pathlib.Path("/home/aaaa/ssss/cccc"),
        pathlib.Path("/home/aaaa/ssss/c"),
        pathlib.Path("/home/aaaa/ssss")
    ),
    (
        pathlib.Path("/aaaa/ssss/cccc"),
        pathlib.Path("/home/aaaa/ssss/c"),
        pathlib.Path("/")
    ),
    (
        pathlib.Path("/aaa/bbbb/prepocess"),
        pathlib.Path("/aaa/bbbb/predict"),
        pathlib.Path("/aaa/bbbb")
    ),
    (
        pathlib.Path("./sample/interim/aaa/bbb"),
        pathlib.Path("./sample"),
        pathlib.Path("./sample")
    ),
])
def test__common_parent(input_dir, output_dir, expect):
    rules = SimlPathRules()
    common_dir = rules.common_parent(
        input_dir,
        output_dir
    )

    assert common_dir == expect

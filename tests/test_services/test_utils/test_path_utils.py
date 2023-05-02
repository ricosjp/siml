import pathlib

import pytest

from siml.utils import path_utils


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
def test__determine_output_directory(
        input_dir, output_dir, str_replace, expect):

    assert path_utils.determine_output_directory(
        input_dir,
        output_dir,
        str_replace
    ) == expect


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
    )
])
def test__common_parent(input_dir, output_dir, expect):
    common_dir = path_utils.common_parent(
        input_dir,
        output_dir
    )

    assert common_dir == expect

import pathlib

from siml.utils import path_utils


def test__determine_output_directory():
    assert path_utils.determine_output_directory(
        pathlib.Path('data/raw/a/b'),
        pathlib.Path('data/sth'),
        'raw'
    ) == pathlib.Path('data/sth/a/b')
    assert path_utils.determine_output_directory(
        pathlib.Path('tests/data/list/data/tet2_3_modulusx0.9000/interim'),
        pathlib.Path('tests/data/list/preprocessed'),
        'interim'
    ) == pathlib.Path(
        'tests/data/list/preprocessed/data/tet2_3_modulusx0.9000'
    )

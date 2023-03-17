import pytest
from siml import prepost
from pathlib import Path


@pytest.mark.parametrize("input_dir, output_dir, expect", [
    (Path("/home/aaaa/ssss/cccc"),
     Path("/home/aaaa/ssss/c"),
     Path("/home/aaaa/ssss")),
    (Path("/aaaa/ssss/cccc"),
     Path("/home/aaaa/ssss/c"),
     Path("/")),
    (Path("/aaa/bbbb/prepocess"),
     Path("/aaa/bbbb/predict"),
     Path("/aaa/bbbb"))
])
def test__common_parent(input_dir, output_dir, expect):
    common_dir = prepost.common_parent(
        input_dir,
        output_dir
    )

    assert common_dir == expect

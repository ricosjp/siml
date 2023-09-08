import pytest

from siml.utils.progress_bar import SimlProgressBar


@pytest.mark.parametrize('total', [
    100, 23, 33
])
def test__is_destroyed_after_iteration_completed(total):
    pbar = SimlProgressBar(total)

    for _ in range(total):
        pbar.update(1)

    assert pbar._pbar is None


@pytest.mark.parametrize('total, desc', [
    (10, "aaaaa"),
    (100, "ccccc"),
    (2, "dddd")
])
def test__consider_inputs(total, desc):
    pbar = SimlProgressBar(total=total, desc=desc)
    pbar._create_pbar()

    assert pbar._pbar.total == total
    assert pbar._pbar.desc == desc


@pytest.mark.parametrize('desc', [
    "aaaa", "loss"
])
def test__consider_desc_when_update(desc):
    pbar = SimlProgressBar(total=100)
    pbar.update(1, desc=desc)
    assert pbar._pbar.desc == desc

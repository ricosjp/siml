
import pathlib

import femio
import numpy as np
import pytest
import scipy.sparse as sp

from siml.preprocessing.converted_objects import (SimlConvertedItem,
                                                  SimlConvertedItemContainer,
                                                  SimlConvertedStatus)


def get_status_item(status: SimlConvertedStatus) -> SimlConvertedItem:
    item = SimlConvertedItem()
    if status == SimlConvertedStatus.successed:
        item.successed()
        return item
    if status == SimlConvertedStatus.failed:
        item.failed()
        return item
    if status == SimlConvertedStatus.skipped:
        item.skipped()
        return item
    if status == SimlConvertedStatus.not_finished:
        return item

    raise NotImplementedError()


def get_status_container(
    n_unfinished: int, n_success: int, n_failed: int, n_skipped: int
) -> SimlConvertedItemContainer:
    values = {}
    for i, n in enumerate([n_unfinished, n_success, n_failed, n_skipped]):
        status = SimlConvertedStatus(i)
        tmp_values = {
            f"{i}_{status.name}": get_status_item(status=status)
            for i in range(n)
        }
        values |= tmp_values

    return SimlConvertedItemContainer(values)


def test__set_skipped():
    item = SimlConvertedItem()
    item.skipped()

    assert item.is_skipped
    assert not item.is_successed
    assert not item.is_failed
    assert item.get_status() == SimlConvertedStatus.skipped.name


def test__set_failed():
    msg = "sample"
    item = SimlConvertedItem()
    item.failed(message=msg)

    assert item.is_failed
    assert not item.is_successed
    assert not item.is_skipped
    assert item.get_status() == SimlConvertedStatus.failed.name


def test__get_failed_message():
    msg = "sample"
    item = SimlConvertedItem()
    item.failed(message=msg)

    assert item.get_failed_message() == msg


def test__set_successed():
    item = SimlConvertedItem()
    item.successed()

    assert item.is_successed
    assert not item.is_failed
    assert not item.is_skipped
    assert item.get_status() == SimlConvertedStatus.successed.name


def test__can_register():
    item = SimlConvertedItem()
    item.successed()
    dict_data = {}
    fem_data = femio.FEMData()
    item.register(dict_data=dict_data, fem_data=fem_data)


@pytest.mark.parametrize("dict_data, fem_data", [
    ({"a": [1, 0]}, femio.FEMData()),
    ({}, None),
    (None, femio.FEMData())
])
def test__cannot_register_multiple_times(dict_data, fem_data):
    item = SimlConvertedItem()
    item.successed()
    item.register(dict_data=dict_data, fem_data=fem_data)

    with pytest.raises(ValueError):
        item.register(dict_data=dict_data, fem_data=fem_data)


@pytest.mark.parametrize('status', [
    SimlConvertedStatus.failed,
    SimlConvertedStatus.skipped,
    SimlConvertedStatus.not_finished
])
def test__cannot_register_when_status_is_not_successed(status):
    item = SimlConvertedItem()
    item._status = status

    with pytest.raises(ValueError):
        item.register(dict_data={}, fem_data=None)


@pytest.mark.parametrize('n_dict_1, n_dict_2', [
    (3, 5), (2, 1), (0, 2)
])
def test__n_items_after_merge_converted_container(n_dict_1, n_dict_2):
    values_1 = {f"{i}_1": SimlConvertedItem() for i in range(n_dict_1)}
    values_2 = {f"{i}_2": SimlConvertedItem() for i in range(n_dict_2)}

    container_1 = SimlConvertedItemContainer(values_1)
    container_2 = SimlConvertedItemContainer(values_2)

    expect = container_1.merge(container_2)
    assert len(expect) == n_dict_1 + n_dict_2


def test__getitem_merge_converted_container():
    n_items = 5
    values_1 = {f"{i}_1": SimlConvertedItem() for i in range(n_items)}
    container = SimlConvertedItemContainer(values_1)

    for k, v in values_1.items():
        assert id(container[k]) == id(v)


@pytest.mark.parametrize("n_unfinished, n_success, n_failed, n_skipped", [
    (5, 4, 3, 1),
    (2, 2, 0, 1),
    (0, 0, 0, 1)
])
def test__select_successed(n_unfinished, n_success, n_failed, n_skipped):
    container = get_status_container(
        n_unfinished, n_success, n_failed, n_skipped
    )

    successed_items = container.select_successed_items()
    assert len(successed_items) == n_success


@pytest.mark.parametrize("n_unfinished, n_success, n_failed, n_skipped", [
    (5, 4, 3, 1),
    (2, 2, 0, 1),
    (0, 0, 0, 1)
])
def test__select_non_successed(n_unfinished, n_success, n_failed, n_skipped):
    container = get_status_container(
        n_unfinished, n_success, n_failed, n_skipped
    )

    items = container.select_non_successed_items()
    assert len(items) == n_unfinished + n_failed + n_skipped


@pytest.mark.parametrize("n_unfinished, n_success, n_failed, n_skipped", [
    (5, 4, 3, 1),
    (2, 2, 0, 1),
    (0, 0, 0, 1)
])
def test__get_keys_incontainer(n_unfinished, n_success, n_failed, n_skipped):
    container = get_status_container(
        n_unfinished, n_success, n_failed, n_skipped
    )

    items = container.select_non_successed_items()
    assert len(items) == n_unfinished + n_failed + n_skipped


@pytest.mark.parametrize("n_unfinished, n_success, n_failed, n_skipped", [
    (5, 4, 3, 1),
    (2, 2, 0, 1),
    (0, 0, 0, 1)
])
def test__query_num_status_items(n_unfinished, n_success, n_failed, n_skipped):
    container = get_status_container(
        n_unfinished, n_success, n_failed, n_skipped
    )

    assert container.query_num_status_items('not_finished') == n_unfinished
    assert container.query_num_status_items('successed') == n_success
    assert container.query_num_status_items('failed') == n_failed
    assert container.query_num_status_items('skipped') == n_skipped
    assert container.query_num_status_items(
        'not_finished', 'successed'
    ) == n_unfinished + n_success
    assert container.query_num_status_items(
        'not_finished', 'successed', 'skipped'
    ) == n_unfinished + n_success + n_skipped
    assert container.query_num_status_items(
        'not_finished', 'successed', 'failed'
    ) == n_unfinished + n_success + n_failed
    assert container.query_num_status_items(
        'not_finished', 'successed', 'failed', 'skipped'
    ) == n_unfinished + n_success + n_failed + n_skipped


def test__raise_error_unknown_status():
    container = SimlConvertedItemContainer({})

    with pytest.raises(ValueError):
        container.query_num_status_items('NONE')


def write_interim_data(directory: pathlib.Path):
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "a.npy", np.random.rand(10, 3))
    sp.save_npz(directory / "s.npz", sp.eye(10))


def test__siml_converted_item_from_interim_directory():
    directory = pathlib.Path("tests/data/tmp/converted_item/interim/0")
    write_interim_data(directory)
    converted_item = SimlConvertedItem.from_interim_directory(directory)
    np.testing.assert_array_equal(
        converted_item._dict_data['a'].shape, (10, 3)
    )
    np.testing.assert_array_equal(
        converted_item._dict_data['s'].shape, (10, 10)
    )


def test__siml_converted_item_container_from_interim_directory():
    directories = [
        pathlib.Path("tests/data/tmp/converted_item/interim/0"),
        pathlib.Path("tests/data/tmp/converted_item/interim/1"),
    ]
    [write_interim_data(d) for d in directories]
    converted_item = SimlConvertedItemContainer.from_interim_directories(
        directories)
    assert len(converted_item) == 2

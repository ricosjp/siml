import pytest

from siml.services.training.logging_items import (
    LoggingFloatItem,
    LoggingIntItem,
    LoggingStrItem,
    LoggingDictItem,
    create_logitems
)


@pytest.mark.parametrize("val, title, expected", [
    (3, "sample", "3"),
    (100, "sample", "100")
])
def test__logging_int_item_to_str(val, title, expected):
    item = LoggingIntItem(val=val, title=title)
    assert item.format() == expected


@pytest.mark.parametrize("val, title, margin, expected", [
    (3, "sample", 0, "3     "),
    (3, "sample", 5, "3          "),
    (100, "sample_ans", 0, "100       ")
])
def test__logging_int_item_to_format_str(val, title, margin, expected):
    item = LoggingIntItem(val=val, title=title)
    assert item.format(padding_margin=margin) == expected


@pytest.mark.parametrize("val, title, formatter", [
    (3.2, "sample", ".2e"),
    (123456.9, "sample", ".2f"),
    (0.0000123456, "sample", ".5e"),
])
def test__logging_float_item_to_str(val, title, formatter):
    item = LoggingFloatItem(val=val, title=title)
    actual = item.format(formatter=formatter)
    expected = f"{val:{formatter}}"
    assert actual == expected


@pytest.mark.parametrize("val, title, formatter, margin", [
    (3.2, "sample", ".2e", 5),
    (123456.9, "sample_v", ".2f", 10),
    (0.0000123456, "val", ".5e", 3),
])
def test__logging_float_item_to_format_str(val, title, formatter, margin):
    item = LoggingFloatItem(val=val, title=title)
    actual = item.format(formatter=formatter, padding_margin=margin)
    expected = f"{val:{formatter}}"
    expected += " " * (max([0, len(title) + margin - len(expected)]))
    assert actual == expected


@pytest.mark.parametrize("val, expected", [
    ("sample", "sample"),
    ("aaaaa", "aaaaa"),
    ("sample_test", "sample_test"),
])
def test__logging_str_item_to_str(val, expected):
    item = LoggingStrItem(val=val)
    actual = item.format()
    assert actual == expected


@pytest.mark.parametrize("val, margin", [
    ("sample", 11),
    ("aaaaa", 5),
    ("sample_test", 4),
])
def test__logging_str_item_to_format_str(val, margin):
    item = LoggingStrItem(val=val)
    actual = item.format(padding_margin=margin)
    expected = val + " " * margin
    assert actual == expected


@pytest.mark.parametrize("val, title, formatter", [
    ({"v1": 3.1415, "v2": 2.7182}, "numerics/", ".3f"),
    ({"v1": 1234455, "v2": 7832989}, "samples/", ".2e"),
])
def test__logging_dict_item_to_str(val, title, formatter):
    item = LoggingDictItem(val=val, title=title)
    actual = item.format(formatter=formatter)

    expected = [f"{v:{formatter}}" for k, v in val.items()]
    expected = ", ".join(expected)
    assert actual == expected


@pytest.mark.parametrize("val, title, formatter, margin", [
    ({"v1": 3.1415, "v2": 2.7182}, "numerics/", ".3f", 5),
    ({"v1": 1234455, "v2": 7832989}, "samples/", ".2e", 8),
])
def test__logging_dict_item_to_format_str(val, title, formatter, margin):
    item = LoggingDictItem(val=val, title=title)
    actual = item.format(formatter=formatter, padding_margin=margin)

    expected = [
        f"{v:{formatter}}".ljust(len(k + title) + margin)
        for k, v in val.items()
    ]
    expected = "".join(expected)
    assert actual == expected


@pytest.mark.parametrize("val, title, formatter, margin", [
    ({}, "numerics/", ".3f", 5),
    ({}, "samples/", ".2e", 8),
])
def test__logging_empty_dict_item_to_format_str(val, title, formatter, margin):
    item = LoggingDictItem(val=val, title=title)
    actual = item.format(formatter=formatter, padding_margin=margin)
    assert actual == ""


@pytest.mark.parametrize("val, title, expected", [
    ("sample", "sample", LoggingStrItem),
    (3, "sample", LoggingIntItem),
    (3.4, "sample", LoggingFloatItem),
    ({"a": 3.14}, "test", LoggingDictItem)
])
def test__create_logitems(val, title, expected):
    item = create_logitems(value=val, title=title)

    assert isinstance(item, expected)

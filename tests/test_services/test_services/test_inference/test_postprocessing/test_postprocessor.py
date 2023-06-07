import pytest

from siml.setting import MainSetting
from siml.services.inference import InnerInfererSetting
from siml.services.inference.postprocessing import PostProcessor


def test__cannot_initialize_with_no_scalers():
    main_setting = MainSetting()
    main_setting.inferer.perform_inverse = True
    inner_setting = InnerInfererSetting(main_setting=main_setting)
    with pytest.raises(ValueError):
        PostProcessor(inner_setting)


@pytest.fixture
def create_postprocessor():
    main_setting = MainSetting()
    main_setting.inferer.perform_inverse = False
    inner_setting = InnerInfererSetting(main_setting=main_setting)
    return PostProcessor(inner_setting)


@pytest.mark.parametrize("dict_data, expect", [
    ({"x": 0}, {"x": 0}),
    ({"x": {"val": 0}}, {"val": 0}),
    ({"x": {"val": 0, "val1": 1}}, {"val": 0, "val1": 1}),
    (None, None),
    ({}, None)
])
def test__formaat_dict_shape(dict_data, expect, create_postprocessor):
    postprocessor: PostProcessor = create_postprocessor
    actual = postprocessor._format_dict_shape(dict_data)

    if expect is not None:
        assert actual == expect
    else:
        assert actual is None

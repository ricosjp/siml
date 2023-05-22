import pathlib
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


@pytest.mark.parametrize("write_simulation_base, expect", [
    (None, True),
    (pathlib.Path("./path_to_not_existed"), True),
    # example of existed path. maybe not approproate
    #  path for write_simulation_base
    (pathlib.Path("tests/data/linear/interim"), False)
])
def test__is_skip_fem_data(
        write_simulation_base, expect, create_postprocessor):
    postprocessor: PostProcessor = create_postprocessor

    assert postprocessor._is_skip_fem_data(write_simulation_base) == expect


@pytest.mark.parametrize("write_simulation_base", [
    (None),
    (pathlib.Path("./tests/sample"))
])
def test__is_skip_fem_data_when_skip_True(write_simulation_base):
    main_setting = MainSetting()
    main_setting.inferer.perform_inverse = False
    main_setting.inferer.skip_fem_data_creation = True
    inner_setting = InnerInfererSetting(main_setting=main_setting)
    postprocessor = PostProcessor(inner_setting)

    assert postprocessor._is_skip_fem_data(write_simulation_base)

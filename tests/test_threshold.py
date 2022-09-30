import pytest

from siml.networks.threshold import Threshold
from siml import setting
import numpy as np


def test__default_threshold():
    block_setting = setting.BlockSetting(
        type='threshold'
    )
    layer = Threshold(block_setting)

    np.testing.assert_almost_equal(layer.threshold, 0)
    np.testing.assert_almost_equal(layer.value, 0)


@pytest.mark.parametrize("threshold, value", [
    (3.2, -1),
    (5.1, 0.1),
    (0, 0)
])
def test__allocate_valid_parameter(threshold, value):
    block_setting = setting.BlockSetting(
        type='threshold',
        optional={"threshold": threshold, "value": value}
    )
    layer = Threshold(block_setting)

    np.testing.assert_almost_equal(layer.threshold, threshold)
    np.testing.assert_almost_equal(layer.value, value)


@pytest.mark.parametrize("threshold", [
    (3.1),
    (5.2),
    (0.2)
])
def test__allocate_default_value_parameter(threshold):
    block_setting = setting.BlockSetting(
        type='threshold',
        optional={"threshold": threshold}
    )
    layer = Threshold(block_setting)

    np.testing.assert_almost_equal(layer.threshold, threshold)
    np.testing.assert_almost_equal(layer.value, threshold)

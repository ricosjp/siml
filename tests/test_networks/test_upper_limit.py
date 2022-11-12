import pytest
import torch
import numpy as np

from siml.networks.upper_limit import UpperLimit
from siml import setting


@pytest.mark.parametrize("inputs, max_value, expects", [
    ([4., 5., 6.], [-1., -1., -1.], [-1., -1., -1.]),
    ([4., 10.1, 12.0], [5., 5., 5.], [4., 5., 5.]),
    ([4., 3.1, 2.0], [5., 5., 5.], [4., 3.1, 2.])
])
def test__upper_limit(inputs, max_value, expects):
    block_setting = setting.BlockSetting(
        type='upper_limit'
    )
    layer = UpperLimit(block_setting)

    x = torch.tensor(inputs).reshape((-1, 1))
    x_max = torch.tensor(max_value).reshape((-1, 1))
    expects = torch.tensor(expects).reshape((-1, 1))

    y = layer.forward(x, x_max)
    np.testing.assert_array_almost_equal(y.numpy(), expects.numpy())

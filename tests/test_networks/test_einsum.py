import pytest
import torch
import numpy as np

from siml.networks.einsum import EinSum
from siml import setting


@pytest.mark.parametrize(
    'equation, a, b', [
        (
            'npqf,nqf->npf',
            np.random.rand(10, 3, 4, 2),
            np.random.rand(10, 4, 2)
        ),
        (
            'npf,mpg->nf',
            np.random.rand(10, 3, 2),
            np.random.rand(1, 3, 1)
        ),
    ])
def test__einsum(equation, a, b):
    block_setting = setting.BlockSetting(
        type='einsum',
        input_names=['a', 'b'],
        optional={'equation': equation},
    )
    layer = EinSum(block_setting)
    actual = layer(torch.from_numpy(a), torch.from_numpy(b))
    expected = np.einsum(equation, a, b)
    np.testing.assert_array_almost_equal(actual.numpy(), expected)

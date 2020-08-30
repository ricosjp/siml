
import numpy as np
import torch

from . import siml_module


class Reducer(siml_module.SimlModule):
    """Broadcastive operation block."""

    def __init__(self, block_setting):
        """Initialize the NN.

        Parameters
        -----------
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
        """
        super().__init__(block_setting, no_parameter=True)

        if 'operator' in block_setting.optional:
            str_op = block_setting.optional['operator']
            if str_op == 'add':
                self.op = torch.add
            elif str_op == 'mul':
                self.op = torch.mul
            else:
                raise ValueError(f"Unknown operator for reducer: {str_op}")
        else:
            self.op = torch.add

        return

    def forward(self, *xs, op=None, supports=None, original_shapes=None):
        if len(xs) == 1:
            return xs[0]

        x = xs[0]
        for other in xs[1:]:
            len_x = len(x.shape)
            len_other = len(other.shape)
            if len_x == len_other:
                x = self.op(x, other)
            elif len_x >= len_other:
                axes = self._get_permute_axis(len_x, len_other)
                x = self.op(x.permute(axes), other)
                x = self._inverse_permute(x, axes)
            else:
                axes = self._get_permute_axis(len_other, len_x)
                x = self.op(x, other.permute(axes))
                x = self._inverse_permute(x, axes)
        return self.activation(x)

    def _get_permute_axis(self, len_x, len_other):
        axes = list(range(len_other - 1, len_x - 1)) \
            + list(range(len_other - 1)) + [len_x - 1]
        return axes

    def _inverse_permute(self, x, axes):
        inverse_axes = np.argsort(axes)
        return x.permute(list(inverse_axes))

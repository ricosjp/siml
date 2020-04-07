
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
                self.op == torch.mul
            else:
                raise ValueError(f"Unknown operator for reducer: {str_op}")
        else:
            self.op = torch.add

        return

    def forward(self, *xs, op=None, supports=None):
        if len(xs) == 1:
            return xs[0]

        x = xs[0]
        for other in xs[1:]:
            x = self.op(x, other)
        return self.activation(x)

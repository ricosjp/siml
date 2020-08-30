
import torch

from . import siml_module


class Reshape(siml_module.SimlModule):
    """Reshape block."""

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        self.new_shape = block_setting.optional['new_shape']
        return

    def forward(self, x, supports=None, original_shapes=None):
        return torch.reshape(x, self.new_shape)

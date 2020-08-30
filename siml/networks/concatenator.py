
import torch

from . import siml_module


class Concatenator(siml_module.SimlModule):
    """Concatenation operation block."""

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        return

    def forward(self, *xs, op=None, supports=None, original_shapes=None):
        return self.activation(torch.cat(xs, dim=-1))

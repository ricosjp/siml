
import torch

from . import header


class Concatenator(header.SimlModule):
    """Concatenation operation block."""

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        return

    def forward(self, *xs, op=None, supports=None):
        return torch.cat(xs, dim=-1)


import torch

from . import header


class Concatenator(torch.nn.Module):
    """Concatenation operation block."""

    def __init__(self, block_setting):
        """Initialize the NN.

        Parameters
        -----------
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
        """
        super().__init__()
        self.activation = header.DICT_ACTIVATIONS[
            block_setting.activations[0]]

        return

    def forward(self, *xs, op=None, supports=None):
        return torch.cat(xs, dim=-1)

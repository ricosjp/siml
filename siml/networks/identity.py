import torch


class Identity(torch.nn.Module):
    """Identity block."""

    def __init__(self, block_setting):
        """Initialize the NN.

        Parameters
        -----------
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
        """
        super().__init__()
        return

    def forward(self, x, supports=None):
        return x

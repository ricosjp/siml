
from . import header
from . import identity


class Activation(identity.Identity):
    """Activation block."""

    def __init__(self, block_setting):
        """Initialize the NN.

        Parameters
        -----------
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
        """
        super().__init__(block_setting)
        if len(block_setting.activations) != 1:
            raise ValueError(
                f"Invalid activation length: {len(block_setting.activations)} "
                f"for {block_setting}")
        self.activation = header.DICT_ACTIVATIONS[
            block_setting.activations[0]]
        return

    def forward(self, x, supports=None):
        return self.activation(x)

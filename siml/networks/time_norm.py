
from . import siml_module


class TimeNorm(siml_module.SimlModule):
    """Normalization for time series data which makes x[t=0, ...] = 0."""

    @staticmethod
    def get_name():
        return 'time_norm'

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return False

    def __init__(self, block_setting):
        """Initialize the NN.

        Parameters
        -----------
        block_setting: siml.setting.BlockSetting
            BlockSetting object.
        """
        super().__init__(block_setting, no_parameter=True)

    def forward(self, x, supports=None, original_shapes=None):
        return self.activation(x - x[[0]])

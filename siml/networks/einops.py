
import einops

from . import siml_module


class Rearrange(siml_module.SimlModule):
    """Apply einops.rearrange."""

    @staticmethod
    def get_name():
        return 'rearrange'

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
        super().__init__(block_setting, no_parameter=True)
        self.pattern = self.block_setting.optional.get('pattern', None)
        if self.pattern is None:
            raise ValueError(f"Feed pattern for: {self.block_setting}")
        self.axes_lengths = self.block_setting.optional.get('axes_lengths', {})
        return

    def forward(self, x, supports=None, original_shapes=None):
        return einops.rearrange(x, pattern=self.pattern, **self.axes_lengths)

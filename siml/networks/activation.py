
from . import siml_module


class Activation(siml_module.SimlModule):
    """Activation block."""

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        self.use_original_shapes = self.block_setting.activations[0] in [
            'max_pool', 'max', 'mean']
        return

    def forward(self, x, supports=None, original_shapes=None):
        if self.use_original_shapes:
            return self.activation(x, original_shapes)
        else:
            return self.activation(x)

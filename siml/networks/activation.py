
from . import siml_module


class Activation(siml_module.SimlModule):
    """Activation block."""

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        return

    def forward(self, x, supports=None):
        return self.activation(x)

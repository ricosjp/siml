from . import header


class Identity(header.SimlModule):
    """Identity block."""

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        return

    def forward(self, x, supports=None):
        return x

import chainer as ch


class Identity(ch.Chain):
    """Identity block."""

    def __init__(self, block_setting):
        """Initialize the NN.

        Args:
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
        """

        super().__init__()

    def __call__(self, x, supports=None):
        return ch.functions.identity(x)

import chainer as ch


class Distributor(ch.Chain):
    """Distributive addition block."""

    def __init__(self, block_setting):
        """Initialize the NN.

        Args:
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
        """
        super().__init__()

    def __call__(self, x0, x1, supports=None):
        return x0 + x1

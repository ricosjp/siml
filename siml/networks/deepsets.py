
from . import AdjustableMLP
from . import header


class DeepSets(header.SimlModule):
    """Permutation equivalent layer published in
    https://arxiv.org/abs/1703.06114 .
    """

    def __init__(self, block_setting):
        """Initialize the NN.

        Parameters
        -----------
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
        """

        super().__init__(block_setting, create_linears=False)
        self.lambda_ = AdjustableMLP(block_setting, last_identity=True)
        self.gamma = AdjustableMLP(block_setting, last_identity=True)

    def _forward_core(self, x, supports=None):
        """Execute the NN's forward computation.

        Parameters
        -----------
            x: numpy.ndarray or cupy.ndarray
                Input of the NN.
            supports: List[chainer.util.CooMatrix]
                List of support inputs.
        Returns
        --------
            y: numpy.ndarray of cupy.ndarray
                Output of the NN.
        """
        h = x
        h = self.activations[-1](
            self.lambda_(h) + header.max_pool(self.gamma(h)))
        return h

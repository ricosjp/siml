import chainer as ch

from . import header


class DeepSets(ch.Chain):
    """Permutation equivalent layer published in
    https://arxiv.org/abs/1703.06114 .
    """

    def __init__(self, block_setting):
        """Initialize the NN.

        Args:
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
        """

        super().__init__()
        nodes = block_setting.nodes
        with self.init_scope():
            self.lambdas = ch.ChainList(*[
                ch.links.Linear(n1, n2)
                for n1, n2 in zip(nodes[:-1], nodes[1:])])
            self.gammas = ch.ChainList(*[
                ch.links.Linear(n1, n2)
                for n1, n2 in zip(nodes[:-1], nodes[1:])])
        self.activations = [
            header.DICT_ACTIVATIONS[activation]
            for activation in block_setting.activations]
        self.dropout_ratios = [
            dropout_ratio for dropout_ratio in block_setting.dropouts]
        self.input_selection = block_setting.input_selection

    def __call__(self, x, supports=None):
        """Execute the NN's forward computation.

        Args:
            x: numpy.ndarray or cupy.ndarray
                Input of the NN.
            supports: List[chainer.util.CooMatrix]
                List of support inputs.
        Returns:
            y: numpy.ndarray of cupy.ndarray
                Output of the NN.
        """
        h = x[:, :, self.input_selection]
        for lambda_, gamma, dropout_ratio, activation in zip(
                self.lambdas, self.gammas,
                self.dropout_ratios, self.activations):
            h = ch.functions.einsum('nmf,gf->nmg', h, lambda_.W) + lambda_.b \
                + ch.functions.max(
                    ch.functions.einsum('nmf,gf->nmg', h, gamma.W) + gamma.b)
            h = ch.functions.dropout(h, ratio=dropout_ratio)
            h = activation(h)
        return h

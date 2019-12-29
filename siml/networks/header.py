import chainer as ch


def max_pool(x):
    return ch.functions.max(x, axis=-2, keepdims=True)


def mean(x):
    return ch.functions.mean(x, axis=-2, keepdims=True)


DICT_ACTIVATIONS = {
    'identity': ch.functions.identity,
    'relu': ch.functions.relu,
    'sigmoid': ch.functions.sigmoid,
    'tanh': ch.functions.tanh,
    'max_pool': max_pool,
    'mean': mean,
}


class AbstractGCN(ch.Chain):

    def __init__(self, block_setting, *, create_subchain=True):
        """Initialize the NN.

        Parameters
        -----------
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
            create_subchain: bool, optional [True]
                If True, create subchain to be trained.
        """

        super().__init__()
        nodes = block_setting.nodes
        with self.init_scope():
            self.subchains = ch.ChainList(*[
                ch.links.Linear(n1, n2)
                for n1, n2 in zip(nodes[:-1], nodes[1:])])
        self.activations = [
            DICT_ACTIVATIONS[activation]
            for activation in block_setting.activations]
        self.dropout_ratios = [
            dropout_ratio for dropout_ratio in block_setting.dropouts]
        self.support_input_index = block_setting.support_input_index
        self.input_selection = block_setting.input_selection

    def __call__(self, x, supports):
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
        hs = ch.functions.stack([
            self._call_single(
                x_[:, self.input_selection],
                supports_[self.support_input_index])
            for x_, supports_ in zip(x, supports)])
        return hs

    def _call_single(self, x, support):
        raise NotImplementedError

import chainer as ch


DICT_ACTIVATIONS = {
    'identity': ch.functions.identity,
    'relu': ch.functions.relu,
    'sigmoid': ch.functions.sigmoid,
    'tanh': ch.functions.tanh,
}


class MLP(ch.ChainList):
    """Multi Layer Perceptron."""

    def __init__(self, block_definition):
        """Initialize MLP object.

        Args:
            unit_numbers: list of int
                List of the number of units for each layer.
            activation_name: str
                The name of the activation function applied to layers except
                for the last one (The activation of the last layer is always
                identity).
            dropout_ratio: float
                The ratio of dropout. Dropout is applied to all layers.
        Returns:
            None
        """
        super().__init__(*[
            ch.links.Linear(unit_number)
            for unit_number in block_definition['nodes']])
        self.activations = [
            DICT_ACTIVATIONS[activation]
            for activation in block_definition['activations']]
        self.dropout_ratios = [
            dropout_ratio for dropout_ratio in block_definition['dropouts']]

    def __call__(self, x):
        """Execute the NN's forward computation.

        Args:
            x: numpy.ndarray or cupy.ndarray
                Input of the NN.
        Returns:
            y: numpy.ndarray or cupy.ndarray
                Output of the NN.
        """
        h = x
        for link, dropout_ratio, activation in zip(
                self, self.dropout_ratios, self.activations):
            h = link(h)
            h = ch.functions.dropout(h, ratio=dropout_ratio)
            h = activation(h)
        return h


class AdjustableMLP(ch.ChainList):
    """Multi Layer Perceptron which accepts arbitray number of dimension. It
    maps (n, m, f) shaped data to (n, m, g) shaped data, where n, m, f, and g
    are sample size, dimension, feature, and converted feature,
    respectively."""

    def __init__(self, block_setting):
        """Initialize the NN.

        Args:
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
        """

        nodes = block_setting.nodes
        super().__init__(*[
            ch.links.Linear(n1, n2)
            for n1, n2 in zip(nodes[:-1], nodes[1:])])
        self.activations = [
            DICT_ACTIVATIONS[activation]
            for activation in block_setting.activations]
        self.dropout_ratios = [
            dropout_ratio for dropout_ratio in block_setting.dropouts]

    def __call__(self, x):
        """Execute the NN's forward computation.

        Args:
            x: numpy.ndarray or cupy.ndarray
                Input of the NN.
        Returns:
            y: numpy.ndarray of cupy.ndarray
                Output of the NN.
        """
        h = x
        for link, dropout_ratio, activation in zip(
                self, self.dropout_ratios, self.activations):
            h = ch.functions.einsum('nmf,gf->nmg', h, link.W) + link.b
            h = ch.functions.dropout(h, ratio=dropout_ratio)
            h = activation(h)
        return h


class Network(ch.ChainList):

    DICT_BLOCKS = {
        'mlp': MLP,
        'adjustable_mlp': AdjustableMLP,
    }

    def __init__(self, model_setting):
        super().__init__(*[
            self.DICT_BLOCKS[block.name](block)
            for block in model_setting.blocks])

    def __call__(self, x):
        h = x
        for link in self:
            h = link(h)
        return h

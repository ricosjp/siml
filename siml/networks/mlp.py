import chainer as ch

from . import header


class MLP(ch.ChainList):
    """Multi Layer Perceptron."""

    def __init__(self, block_setting):
        """Initialize MLP object.

        Parameters
        -----------
            unit_numbers: list of int
                List of the number of units for each layer.
            activation_name: str
                The name of the activation function applied to layers except
                for the last one (The activation of the last layer is always
                identity).
            dropout_ratio: float
                The ratio of dropout. Dropout is applied to all layers.
        Returns
        --------
            None
        """

        nodes = block_setting.nodes
        super().__init__(*[
            ch.links.Linear(n1, n2)
            for n1, n2 in zip(nodes[:-1], nodes[1:])])
        self.activations = [
            header.DICT_ACTIVATIONS[activation]
            for activation in block_setting.activations]
        self.dropout_ratios = [
            dropout_ratio for dropout_ratio in block_setting.dropouts]

    def __call__(self, x, supports=None):
        """Execute the NN's forward computation.

        Parameters
        -----------
            x: numpy.ndarray or cupy.ndarray
                Input of the NN.
        Returns
        --------
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

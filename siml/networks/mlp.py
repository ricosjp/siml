
import torch.nn.functional as functional

from . import header


class MLP(header.AbstractMLP):
    """Multi Layer Perceptron."""

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
        h = x[:, :, self.input_selection]
        for linear, dropout_ratio, activation in zip(
                self.linears, self.dropout_ratios, self.activations):
            h = linear(h)
            h = functional.dropout(h, p=dropout_ratio, training=self.training)
            h = activation(h)
        return h

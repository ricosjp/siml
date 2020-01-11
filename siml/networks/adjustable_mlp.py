
import torch
import torch.nn.functional as functional

from . import header


class AdjustableMLP(header.AbstractMLP):
    """Multi Layer Perceptron which accepts arbitray number of dimension. It
    maps (n, m, f) shaped data to (n, m, g) shaped data, where n, m, f, and g
    are sample size, dimension, feature, and converted feature,
    respectively."""

    def __call__(self, x, supports=None):
        """Execute the NN's forward computation.

        Parameters
        -----------
            x: numpy.ndarray or cupy.ndarray
                Input of the NN.
        Returns
        --------
            y: numpy.ndarray of cupy.ndarray
                Output of the NN.
        """
        h = x[:, :, self.input_selection]
        for linear, dropout_ratio, activation in zip(
                self.linears, self.dropout_ratios, self.activations):
            h = torch.einsum('nmf,gf->nmg', h, linear.W) + linear.b
            h = functional.dropout(h, p=dropout_ratio, training=self.training)
            h = activation(h)
        return h

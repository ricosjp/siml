
import torch
import torch.nn.functional as functional

from . import siml_module


class IdMLP(siml_module.SimlModule):
    """Multi Layer Perceptron with adding the identity."""

    @staticmethod
    def get_name():
        return 'id_mlp'

    @staticmethod
    def is_trainable():
        return True

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return False

    def _forward_core(self, x, supports=None, original_shapes=None):
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
        for linear, dropout_ratio, activation in zip(
                self.linears, self.dropout_ratios, self.activations):
            h = self._apply_linear(linear, h)
            h = functional.dropout(h, p=dropout_ratio, training=self.training)
            h = activation(h)
        return h

    def _apply_linear(self, linear, h):
        w = linear.weight
        h = torch.einsum('n...f,fg->n...g', h, w.T + 1)
        b = linear.bias
        if b is None:
            return h
        else:
            return h + b

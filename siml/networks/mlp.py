
import torch.nn.functional as functional

from . import siml_module


class MLP(siml_module.SimlModule):
    """Multi Layer Perceptron."""

    @staticmethod
    def get_name():
        return 'mlp'

    @staticmethod
    def is_trainable():
        return True

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return False

    def __init__(self, block_setting, **kwargs):
        super().__init__(block_setting, **kwargs)
        self.clone = self.block_setting.optional.get('clone', False)
        return

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
        if self.clone:
            h = x.clone()
        else:
            h = x
        for linear, dropout_ratio, activation in zip(
                self.linears, self.dropout_ratios, self.activations):
            h = linear(h)
            h = functional.dropout(h, p=dropout_ratio, training=self.training)
            h = activation(h)
        if self.clone:
            return h.clone()
        else:
            return h

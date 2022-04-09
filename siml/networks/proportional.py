
import torch
import torch.nn.functional as functional

from . import siml_module


class Proportional(siml_module.SimlModule):
    """Proportional layer."""

    @staticmethod
    def get_name():
        return 'proportional'

    @staticmethod
    def is_trainable():
        return True

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return False

    def __init__(self, block_setting):
        block_setting.bias = False
        super().__init__(block_setting)
        if len(self.block_setting.nodes) != 2:
            raise ValueError(
                f"Proportional layer cannot be deep: {self.block_setting}")
        if self.block_setting.activations[0] != 'identity':
            raise ValueError(
                'Proportional layer cannot have activation: '
                f"{self.block_setting}")
        self.positive_weight = self.block_setting.optional.get(
            'positive_weight', False)
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
        if self.positive_weight:
            h = torch.einsum(
                'n...f,fg->n...g', x, torch.relu(self.linears[0].weight.T))
        else:
            h = torch.einsum(
                'n...f,fg->n...g', x, self.linears[0].weight.T)
        return h

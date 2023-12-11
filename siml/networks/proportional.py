
import torch

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
        str_positive_weight_method = self.block_setting.optional.get(
            'positive_weight_method', 'sigmoid')  # or 'square', 'shifted_tanh'
        if self.positive_weight:
            if str_positive_weight_method == 'sigmoid':
                self.compute_weight = torch.sigmoid
            elif str_positive_weight_method == 'abs':
                self.compute_weight = torch.abs
            elif str_positive_weight_method == 'square':
                self.compute_weight = self._square
            elif str_positive_weight_method == 'shifted_tanh':
                self.compute_weight = self._shifted_tanh
            else:
                raise ValueError(f"Unexpected {str_positive_weight_method = }")
            print(f"positive_weight_method: {str_positive_weight_method}")
        else:
            self.compute_weight = self._id
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
        h = torch.einsum('n...f,fg->n...g', x, self.get_weight().T)
        return h

    def get_weight(self):
        return self.compute_weight(self.linears[0].weight)

    def _id(self, w):
        return w

    def _square(self, w):
        return torch.einsum('ij,ij->ij', w, w)

    def _shifted_tanh(self, w):
        return torch.tanh(w) + 1

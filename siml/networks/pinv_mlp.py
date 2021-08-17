
import torch

from . import activations
from . import siml_module


class PInvMLP(siml_module.SimlModule):
    """Pseudo inverse of Multi Layer Perceptron."""

    def __init__(self, block_setting, reference_block):
        super().__init__(
            block_setting, create_linears=False,
            create_activations=False, create_dropouts=False,
            no_parameter=True)
        self.epsilon = self.block_setting.optional.get('epsilon', 1.e-5)
        self.reference_block = reference_block
        self.linears = [
            PInvLinear(linear)
            for linear in self.reference_block.linears[-1::-1]]
        self.activations = [
            self._define_activation(name)
            for name in self.reference_block.block_setting.activations[-1::-1]]

        return

    def _forward_core(self, x, supports=None, original_shapes=None):
        """Execute the NN's forward computation.

        Parameters
        ----------
        x: torch.Tensor
            Input of the NN.

        Returns
        -------
        y: torch.Tensor
            Output of the NN.
        """
        h = x
        for linear, activation in zip(self.linears, self.activations):
            h = activation(h)  # Activation first, because it is inverse
            h = linear(h)
        return h

    def _define_activation(self, name):
        if name == 'identity':
            return activations.identity
        elif name == 'tanh':
            return activations.ATanh(epsilon=self.epsilon)
        else:
            raise ValueError(f"Unsupported activation name: {name}")


class PInvLinear(torch.nn.Module):

    def __init__(self, ref_linear):
        super().__init__()
        self.ref = ref_linear
        return

    def forward(self, x):
        w = self.weight
        b = self.bias

        h = torch.einsum('n...f,fg->n...g', x - b, torch.pinverse(w.T))
        return h

    @property
    def weight(self):
        return self.ref.weight

    @property
    def bias(self):
        return self.ref.bias

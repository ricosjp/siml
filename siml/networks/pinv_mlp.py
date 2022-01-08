
import torch

from . import activations
from . import id_mlp
from . import mlp
from . import normalized_mlp
from . import siml_module


class PInvMLP(siml_module.SimlModule):
    """Pseudo inverse of Multi Layer Perceptron."""

    @staticmethod
    def get_name():
        return 'pinv_mlp'

    @staticmethod
    def is_trainable():
        return True

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return False

    @classmethod
    def _get_n_input_node(
            cls, block_setting, predecessors, dict_block_setting,
            input_length, **kwargs):
        return dict_block_setting[block_setting.reference_block_name].nodes[-1]

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length, **kwargs):
        return dict_block_setting[block_setting.reference_block_name].nodes[0]

    def __init__(self, block_setting, reference_block):
        super().__init__(
            block_setting, create_linears=False,
            create_activations=False, create_dropouts=False,
            no_parameter=True)
        self.epsilon = self.block_setting.optional.get('epsilon', 1.e-5)
        self.reference_block = reference_block
        if isinstance(
                self.reference_block, normalized_mlp.NormalizedMLP):
            self.option = 'normalized_mlp'
        elif isinstance(self.reference_block, id_mlp.IdMLP):
            self.option = 'id_mlp'
        elif isinstance(self.reference_block, mlp.MLP):
            self.option = 'mlp'
        else:
            raise ValueError(
                f"Unexpected reference block {self.reference_block}"
                f"for {self.block_setting}")
        self.linears = [
            PInvLinear(linear, option=self.option)
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
        elif name == 'leaky_relu':
            return activations.InversedLeakyReLU()
        else:
            raise ValueError(f"Unsupported activation name: {name}")


class PInvLinear(torch.nn.Module):

    def __init__(self, ref_linear, option):
        super().__init__()
        self.ref = ref_linear
        self.option = option
        return

    def forward(self, x):
        if self.option == 'normalized_mlp':
            w = self.weight / torch.max(torch.abs(self.weight))
        elif self.option == 'id_mlp':
            w = self.weight + 1
        elif self.option == 'mlp':
            w = self.weight
        else:
            raise ValueError(f"Unexpected option: {self.option}")
        b = self.bias

        if b is None:
            h = torch.einsum(
                'n...f,fg->n...g', x, torch.pinverse(w.T))
        else:
            h = torch.einsum(
                'n...f,fg->n...g', x - b, torch.pinverse(w.T))
        return h

    @property
    def weight(self):
        return self.ref.weight

    @property
    def bias(self):
        return self.ref.bias

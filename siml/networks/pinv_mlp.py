
import torch

from . import activations
from . import id_mlp
from . import mlp
from . import normalized_mlp
from . import proportional
from . import siml_module
from . import tensor_operations


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
        elif isinstance(
                self.reference_block, tensor_operations.EnSEquivariantMLP):
            self.option = 'ens_equivariant_mlp'
        elif isinstance(self.reference_block, proportional.Proportional):
            self.option = 'proportional'
        else:
            raise ValueError(
                f"Unexpected reference block {self.reference_block}"
                f"for {self.block_setting}")
        if self.option == 'proportional':
            self.linears = [
                PInvLinear(self.reference_block, option=self.option)]
        else:
            self.linears = [
                PInvLinear(linear, option=self.option)
                for linear in self.reference_block.linears[-1::-1]]
        self.activations = [
            self._define_activation(name)
            for name in self.reference_block.block_setting.activations[-1::-1]]
        self.clone = self.block_setting.optional.get('clone', False)

        return

    def _forward_core(self, xs, supports=None, original_shapes=None):
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
        if self.option == 'ens_equivariant_mlp':
            if not isinstance(xs, list):
                raise ValueError(
                    f"Input should be list for: {self.block_setting}")

            x = xs[0]
            length = xs[1][..., 0]
            if len(xs) > 2:
                time = xs[2][0, 0]  # NOTE: Assume global data
            else:
                time = 1
            if len(xs) > 3:
                mass = xs[3][0, 0]  # NOTE: Assume global data
            else:
                mass = 1
            if len(xs) > 4:
                raise ValueError(f"Unexpected input type: {xs}")

            power_length = self.reference_block.power_length
            power_time = self.reference_block.power_time
            power_mass = self.reference_block.power_mass

            raise ValueError(length, time)
            x = torch.einsum(
                'i...,i->i...', x, 1 / length**power_length) \
                / time**power_time \
                / mass**power_mass
        else:
            x = xs

        if self.clone:
            h = x.clone()
        else:
            h = x
        for linear, activation in zip(self.linears, self.activations):
            h = activation(h)  # Activation first, because it is inverse
            h = linear(h)

        if self.option == 'ens_equivariant_mlp':
            h = torch.einsum(
                'i...,i->i...', h, length**power_length) \
                * time**power_time \
                * mass**power_mass

        if self.clone:
            return h.clone()
        else:
            return h

    def _define_activation(self, name):
        if name == 'identity':
            return activations.identity
        elif name == 'tanh':
            return activations.ATanh(epsilon=self.epsilon)
        elif name == 'leaky_relu':
            return activations.InversedLeakyReLU()
        elif name == 'smooth_leaky_relu':
            return activations.inversed_smooth_leaky_relu
        else:
            raise ValueError(f"Unsupported activation name: {name}")


class PInvLinear(torch.nn.Module):

    def __init__(self, ref_linear, option):
        super().__init__()
        self.ref = ref_linear
        self.option = option
        return

    def forward(self, x):
        h = torch.einsum(
            'n...f,fg->n...g', x + self.bias, self.weight.T)
        return h

    @property
    def weight(self):
        """Return pseudo inversed weight."""
        if self.option == 'normalized_mlp':
            w = self.ref.weight / torch.max(torch.abs(self.ref.weight))
        elif self.option == 'id_mlp':
            w = self.ref.weight + 1
        elif self.option in ['mlp', 'ens_equivariant_mlp']:
            w = self.ref.weight
        elif self.option in ['proportional']:
            w = self.ref.get_weight()
        else:
            raise ValueError(f"Unexpected option: {self.option}")
        return torch.pinverse(w)

    @property
    def bias(self):
        """Return inverse bias."""
        if not hasattr(self.ref, 'bias'):
            return 0
        elif self.ref.bias is None:
            return 0
        else:
            return - self.ref.bias

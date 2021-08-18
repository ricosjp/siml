
import torch

from . import activations
from . import siml_module


class Dirichlet(siml_module.SimlModule):
    """Dirichlet boundary condition management."""

    @staticmethod
    def get_name():
        return 'dirichlet'

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def accepts_multiple_inputs():
        return True

    @staticmethod
    def uses_support():
        return False

    @classmethod
    def _get_n_input_node(
            cls, block_setting, predecessors, dict_block_setting,
            input_length):
        return dict_block_setting[predecessors[0]].nodes[-1]

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length):
        return input_node

    def __init__(self, block_setting):
        """Initialize the module.

        Parameters
        -----------
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
        """
        super().__init__(
            block_setting, no_parameter=True, create_activations=False)
        return

    def forward(
            self, *xs, supports=None, original_shapes=None):
        """
        Take into account Dirichlet boundary condition.

        Parameters
        ----------
        xs: List[torch.Tensor]
            0: Variable values
            1: Dirichlet values.

        Returns
        -------
        ys: torch.Tensor
            Variable values with Dirichlet.
        """
        if len(xs) != 2:
            raise ValueError(
                f"Input should be x and Dirichlet ({len(xs)} given)")
        x = xs[0]
        dirichlet = xs[1]
        filter_not_nan = ~ torch.isnan(dirichlet)
        x[filter_not_nan] = dirichlet[filter_not_nan]
        return x


class NeumannIsoGCN(siml_module.SimlModule):
    """Neumann boundary condition management using IsoGCN."""

    @staticmethod
    def get_name():
        return 'neumann_isogcn'

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def accepts_multiple_inputs():
        return True

    @staticmethod
    def uses_support():
        return False

    def __init__(self, block_setting):
        """Initialize the module.

        Parameters
        -----------
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
        """
        super().__init__(
            block_setting, no_parameter=True, create_activations=False)
        return

    def forward(
            self, *xs, supports=None, original_shapes=None):
        """
        Take into account Neumann boundary condition using IsoGCN.

        Parameters
        ----------
        xs: List[torch.Tensor]
            0: Gradient values without Neumann.
            1: Neumann values multiplied with normal vectors.
            2: Inversed moment matrices.

        Returns
        -------
        ys: torch.Tensor
            Gradient values with Neumann.
        """
        if len(xs) != 3:
            raise ValueError(
                f"Input shoulbe x and Dirichlet ({len(xs)} given)")
        grad = xs[0]
        directed_neumann = xs[1]
        inversed_moment_tensors = xs[2]
        return grad + torch.einsum(
            'ikl,ilf->ikf',
            inversed_moment_tensors[..., 0], directed_neumann)


class NeumannEncoder(siml_module.SimlModule):
    """Encoder for Neumann condition."""

    @staticmethod
    def get_name():
        return 'neumann_encoder'

    @staticmethod
    def is_trainable():
        return True

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return False

    def __init__(self, block_setting, reference_block):
        super().__init__(
            block_setting, create_linears=False,
            create_activations=False, create_dropouts=False,
            no_parameter=True)
        self.epsilon = self.block_setting.optional.get('epsilon', 1.e-5)
        self.reference_block = reference_block
        self.activation_names = [
            name for name in self.reference_block.block_setting.activations]
        self.derivative_activations = [
            self._define_activation_derivative(name)
            for name in self.activation_names]
        return

    @property
    def linears(self):
        return self.reference_block.linears

    @property
    def weights(self):
        return [linear.weight for linear in self.linears]

    @property
    def activations(self):
        return self.reference_block.activations

    def forward(
            self, *xs, supports=None, original_shapes=None):
        """
        Take into account Neumann boundary condition using IsoGCN.

        Parameters
        ----------
        xs: List[torch.Tensor]
            0: Variable values
            1: Neumann values multiplied with normal vectors.

        Returns
        -------
        ys: torch.Tensor
            Embedded Neumann values multiplied with normal vectors.
        """
        if len(xs) != 2:
            raise ValueError(
                f"Input shoulbe x and Neumann ({len(xs)} given)")
        h = xs[0]
        directed_neumann = xs[1]
        for linear, name, activation, derivative_activation in zip(
                self.linears, self.activation_names,
                self.activations, self.derivative_activations):
            lineared_h = linear(h)
            if name == 'identity':
                directed_neumann = torch.einsum(
                    'i...f,fg->i...g', directed_neumann, linear.weight.T)
            else:
                derivative_h = derivative_activation(lineared_h)
                directed_neumann = torch.einsum(
                    'ig,i...g->i...g', derivative_h, torch.einsum(
                        'i...f,fg->i...g', directed_neumann, linear.weight.T))
            h = activation(lineared_h)

        return directed_neumann

    def _define_activation_derivative(self, name):
        if name == 'identity':
            return activations.one
        elif name == 'tanh':
            return activations.derivative_tanh
        else:
            raise ValueError(f"Unsupported activation name: {name}")

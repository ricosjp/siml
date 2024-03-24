
import numpy as np
import torch

from .. import setting
from . import activations
from . import proportional
from . import mlp
from . import siml_module
from . import reducer


class Contraction(siml_module.SimlModule):
    """Contraction block."""

    @staticmethod
    def get_name():
        return 'contraction'

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
            input_length, **kwargs):
        return np.sum([
            dict_block_setting[predecessor].nodes[-1]
            for predecessor in predecessors])

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length, **kwargs):
        return np.max([
            dict_block_setting[predecessor].nodes[-1]
            for predecessor in predecessors])

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        return

    def forward(self, *xs, supports=None, original_shapes=None):
        """Calculate tensor contraction of rank n ( > m) and m tensors
        \\sum_{l_1, ..., l_m}
        A_{i,k_1,k_2,...,l_1,l_2,...,l_{m}} B_{i,l_1,l_2,...,l_m}
        """
        if len(xs) == 1:
            x = xs[0]
            y = xs[0]
        elif len(xs) == 2:
            x = xs[0]
            y = xs[1]
        else:
            raise ValueError(f"1 or 2 inputs expected. Given: {len(xs)}")
        rank_x = len(x.shape) - 2  # [n_vertex, dim, dim, ..., n_feature]
        rank_y = len(y.shape) - 2  # [n_vertex, dim, dim, ..., n_feature]
        if rank_x < rank_y:
            # Force make rank x has the same or higher rank
            x, y = y, x
            rank_x, rank_y = rank_y, rank_x
        string_x = 'abcdefghijklmnopqrstuvwxy'[:1+rank_x] + 'z'
        string_y = 'a' + string_x[-1-rank_y:-1] + 'z'
        rank_diff = rank_x - rank_y
        string_res = string_x[:1+rank_diff] + 'z'
        return self.activation(
            torch.einsum(f"{string_x},{string_y}->{string_res}", x, y))


class TensorProduct(siml_module.SimlModule):
    """Tensor product block."""

    @staticmethod
    def get_name():
        return 'tensor_product'

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
            input_length, **kwargs):
        return np.sum([
            dict_block_setting[predecessor].nodes[-1]
            for predecessor in predecessors])

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length, **kwargs):
        return np.max([
            dict_block_setting[predecessor].nodes[-1]
            for predecessor in predecessors])

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        return

    def forward(self, *xs, supports=None, original_shapes=None):
        """Calculate tensor product of rank n and m tensors
        A_{i,k_1,k_2,...,k_m} B_{i,l_1,l_2,...,l_m}
        """
        if len(xs) == 1:
            x = xs[0]
            y = xs[0]
        elif len(xs) == 2:
            x = xs[0]
            y = xs[1]
        else:
            raise ValueError(f"1 or 2 inputs expected. Given: {len(xs)}")
        rank_x = len(x.shape) - 2  # [n_vertex, dim, dim, ..., n_feature]
        rank_y = len(y.shape) - 2  # [n_vertex, dim, dim, ..., n_feature]
        original_string = 'abcdefghijklmnopqrstuvwxy'
        string_x = original_string[:1+rank_x] + 'z'
        string_y = 'a' + original_string[1+rank_x:1+rank_x+rank_y] + 'z'
        string_res = original_string[:1+rank_x+rank_y] + 'z'
        return self.activation(
            torch.einsum(f"{string_x},{string_y}->{string_res}", x, y))


class EquivariantMLP(siml_module.SimlModule):
    """E(n) equivariant MLP block."""

    @staticmethod
    def get_name():
        return 'equivariant_mlp'

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
        super().__init__(block_setting)

        self.mul = reducer.Reducer(
            setting.BlockSetting(optional={'operator': 'mul'}))
        self.create_linear_weight = self.block_setting.optional.get(
            'create_linear_weight', False)
        self.positive = self.block_setting.optional.get(
            'positive', False)
        self.positive_weight_method = self.block_setting.optional.get(
            'positive_weight_method', 'sigmoid')
        self.normalize = self.block_setting.optional.get(
            'normalize', False)
        self.sqrt = self.block_setting.optional.get('sqrt', False)
        if block_setting.nodes[0] == block_setting.nodes[-1] and \
                not self.create_linear_weight:
            self.linear_weight = activations.identity
        else:
            self.linear_weight = proportional.Proportional(
                setting.BlockSetting(
                    nodes=[
                        block_setting.nodes[0],
                        block_setting.nodes[-1],
                    ],
                    activations=['identity'],
                    optional={
                        'positive_weight': self.positive,
                        'positive_weight_method': self.positive_weight_method,
                    },
                ))

        if self.positive:
            self.filter_coeff = torch.abs
        else:
            self.filter_coeff = activations.identity
        self.contraction = Contraction(setting.BlockSetting())

        if self.residual:
            self.residual_weight = self.block_setting.optional.get(
                'residual_weight', 0.5)
            print(f"Residual EqMLP with coeff: {self.residual_weight}")

        self.invariant = block_setting.optional.get('invariant', False)
        if self.invariant:
            print(f"Invariant for {block_setting.name}")
        return

    def _forward(self, x, supports=None, original_shapes=None):
        # NOTE: We define _forward instead of _forward_core to manage
        #       residual connection by itself.
        h = self._forward_core(
            x, supports=supports, original_shapes=original_shapes)
        return h

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
        h = self.contraction(x)
        if self.normalize:
            x = torch.einsum('n...f,nf->n...f', x, 1 / torch.sqrt(h + 1e-5))
        if self.sqrt:
            h = torch.sqrt(h + 1e-5)  # To avoid infinite gradient
        if self.residual:
            original_h = torch.clone(h)

        linear_x = self.linear_weight(x)
        for linear, dropout_ratio, activation in zip(
                self.linears, self.dropout_ratios, self.activations):
            h = linear(h)
            h = torch.nn.functional.dropout(
                h, p=dropout_ratio, training=self.training)

            h = activation(h)

        if self.residual:
            h = (1 - self.residual_weight) * h \
                + self.residual_weight * self.shortcut(original_h)

        if self.invariant:
            return self.filter_coeff(h)
        else:
            return torch.einsum(
                'i...f,if->i...f', linear_x, self.filter_coeff(h))


class EnSEquivariantMLP(EquivariantMLP):

    @staticmethod
    def get_name():
        return 'ens_equivariant_mlp'

    @staticmethod
    def accepts_multiple_inputs():
        return True

    def __init__(self, block_setting):
        super().__init__(block_setting)
        if 'dimension' not in block_setting.optional:
            raise ValueError(f"Set optional.dimension for: {block_setting}")

        self.diff = block_setting.optional.get('diff', True)
        if self.diff:
            print(f"diff mode for {block_setting.name}")
        dimension = block_setting.optional['dimension']
        self.power_length = dimension['length']
        self.power_time = dimension['time']
        self.power_mass = dimension['mass']

        self.show_scale = block_setting.optional.get('show_scale', False)
        self.invariant = block_setting.optional.get('invariant', False)
        if self.invariant:
            print(f"Invariant for {block_setting.name}")

        return

    def _forward_core(
            self, xs, supports=None, original_shapes=None,
            power_length=None, power_time=None, power_mass=None):
        """Execute the NN's forward computation.

        Parameters
        -----------
        xs: list[torch.Tensor]
            - 0: Input of the NN.
            - 1: Length scales.
            - 2: Time scales.
            - 3: Mass scales.

        Returns
        --------
        y: torch.Tensor
            Output of the NN.
        """
        if len(xs) < 2:
            raise ValueError(f"Feed dimension data for: {self.block_setting}")

        x = xs[0]
        length = xs[1]
        if len(xs) > 2:
            time = xs[2][0, 0]  # NOTE: Assume global data
        else:
            time = 1
        if len(xs) > 3:
            mass = xs[3]
            if mass is None:
                mass = 1
        else:
            mass = 1
        if len(xs) > 4:
            raise ValueError(f"Unexpected input type: {xs}")

        if power_length is None:
            power_length = self.power_length
        if power_time is None:
            power_time = self.power_time
        if power_mass is None:
            power_mass = self.power_mass

        if isinstance(mass, torch.Tensor):
            x = torch.einsum(
                'i...,ia,ia->i...', x,
                1 / length**power_length, 1 / mass**power_mass) \
                / time**power_time
        else:
            x = torch.einsum(
                'i...,ia->i...', x, 1 / length**power_length) \
                / time**power_time \
                / mass**power_mass

        if self.diff:
            volume = length**3
            mean = torch.sum(
                torch.einsum('i...,ia->i...', x, volume),
                dim=0, keepdim=True) / torch.sum(volume)
            x = x - mean

        if self.show_scale:
            x_norm = torch.einsum('...,...->', x, x)**.5
            print(f"{self.block_setting.name}: {x_norm}")
        h = super()._forward_core(x)

        if self.diff:
            linear_mean = self.linear_weight(mean)
            h = h + linear_mean

        if self.invariant:
            return h
        else:
            if isinstance(mass, torch.Tensor):
                h = torch.einsum(
                    'i...,ia,ia->i...', h,
                    length**power_length, mass**power_mass) * time**power_time
            else:
                h = torch.einsum(
                    'i...,ia->i...', h, length**power_length) \
                    * time**power_time \
                    * mass**power_mass
            return h


class SEquivariantMLP(mlp.MLP):

    @staticmethod
    def get_name():
        return 's_equivariant_mlp'

    @staticmethod
    def accepts_multiple_inputs():
        return True

    def __init__(self, block_setting):
        super().__init__(block_setting)

        self.create_linear_weight = self.block_setting.optional.get(
            'create_linear_weight', False)
        if block_setting.nodes[0] == block_setting.nodes[-1] and \
                not self.create_linear_weight:
            self.linear_weight = activations.identity
        else:
            self.linear_weight = proportional.Proportional(
                setting.BlockSetting(
                    nodes=[
                        block_setting.nodes[0],
                        block_setting.nodes[-1],
                    ],
                    activations=['identity'],
                    optional={
                        'positive_weight': self.positive,
                        'positive_weight_method': self.positive_weight_method,
                    },
                ))

        if 'dimension' not in block_setting.optional:
            raise ValueError(f"Set optional.dimension for: {block_setting}")

        self.diff = block_setting.optional.get('diff', True)
        if self.diff:
            print(f"diff mode for {block_setting.name}")
        dimension = block_setting.optional['dimension']
        self.power_length = dimension['length']
        self.power_time = dimension['time']
        self.power_mass = dimension['mass']

        self.show_scale = block_setting.optional.get('show_scale', False)
        self.invariant = block_setting.optional.get('invariant', False)
        if self.invariant:
            print(f"Invariant for {block_setting.name}")

        return

    def _forward_core(
            self, xs, supports=None, original_shapes=None,
            power_length=None, power_time=None, power_mass=None):
        """Execute the NN's forward computation.

        Parameters
        -----------
        xs: list[torch.Tensor]
            - 0: Input of the NN.
            - 1: Length scales.
            - 2: Time scales.
            - 3: Mass scales.

        Returns
        --------
        y: torch.Tensor
            Output of the NN.
        """
        if len(xs) < 2:
            raise ValueError(f"Feed dimension data for: {self.block_setting}")

        x = xs[0]
        length = xs[1]
        if len(xs) > 2:
            time = xs[2][0, 0]  # NOTE: Assume global data
        else:
            time = 1
        if len(xs) > 3:
            mass = xs[3]
            if mass is None:
                mass = 1
        else:
            mass = 1
        if len(xs) > 4:
            raise ValueError(f"Unexpected input type: {xs}")

        if power_length is None:
            power_length = self.power_length
        if power_time is None:
            power_time = self.power_time
        if power_mass is None:
            power_mass = self.power_mass

        if isinstance(mass, torch.Tensor):
            x = torch.einsum(
                'i...,ia,ia->i...', x,
                1 / length**power_length, 1 / mass**power_mass) \
                / time**power_time
        else:
            x = torch.einsum(
                'i...,ia->i...', x, 1 / length**power_length) \
                / time**power_time \
                / mass**power_mass

        if self.diff:
            volume = length**3
            mean = torch.sum(
                torch.einsum('i...,ia->i...', x, volume),
                dim=0, keepdim=True) / torch.sum(volume)
            x = x - mean

        if self.show_scale:
            x_norm = torch.einsum('...,...->', x, x)**.5
            print(f"{self.block_setting.name}: {x_norm}")
        h = super()._forward_core(x)

        if self.diff:
            linear_mean = self.linear_weight(mean)
            h = h + linear_mean

        if self.invariant:
            return h
        else:
            if isinstance(mass, torch.Tensor):
                h = torch.einsum(
                    'i...,ia,ia->i...', h,
                    length**power_length, mass**power_mass) * time**power_time
            else:
                h = torch.einsum(
                    'i...,ia->i...', h, length**power_length) \
                    * time**power_time \
                    * mass**power_mass
            return h


class SharedEnSEquivariantMLP(siml_module.SimlModule):
    """
    EnSEquivariantMLP layer tied with the reference EnSEquivariantMLP.
    Useful for emcoders of Neumann conditions.
    """

    @staticmethod
    def get_name():
        return 'shared_ens_equivariant_mlp'

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
        return dict_block_setting[block_setting.reference_block_name].nodes[0]

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length, **kwargs):
        return dict_block_setting[block_setting.reference_block_name].nodes[-1]

    def __init__(self, block_setting, reference_block):
        super().__init__(
            block_setting, create_linears=False,
            create_activations=False, create_dropouts=False,
            no_parameter=True)
        self.reference_block = reference_block

        if 'dimension' not in block_setting.optional:
            raise ValueError(f"Set optional.dimension for: {block_setting}")
        dimension = block_setting.optional['dimension']
        self.power_length = dimension['length']
        self.power_time = dimension['time']
        self.power_mass = dimension['mass']

        return

    @property
    def linears(self):
        return self.reference_block.linears

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
        return self.reference_block(xs)


class ConservativeEnSEquivariantMLP(EnSEquivariantMLP):

    @staticmethod
    def get_name():
        return 'cons_ens_equivariant_mlp'

    @staticmethod
    def accepts_multiple_inputs():
        return True

    def __init__(self, block_setting):
        super().__init__(block_setting)
        self.spatial_weight = self.block_setting.optional.get(
            'spatial_weight', True)
        return

    def _forward_core(
            self, xs, supports=None, original_shapes=None,
            power_length=None, power_time=None, power_mass=None):
        """Execute the NN's forward computation.

        Parameters
        -----------
        xs: list[torch.Tensor]
            - 0: Input of the NN.
            - 1: Length scales.
            - 2: Time scales.
            - 3: Mass scales.

        Returns
        --------
        y: torch.Tensor
            Output of the NN.
        """
        x = xs[0]
        cons_x = self.integral(x, xs[1])
        h = super()._forward_core(xs)
        cons_h = self.integral(h, xs[1])
        return torch.einsum('...,i...->i...', cons_x / cons_h, h)

    def integral(self, x, length_scale):
        if self.spatial_weight:
            weight = length_scale**3  # Volume
            return torch.einsum('ia,i...->...', weight, x)
        else:
            return torch.sum(x, dim=0)

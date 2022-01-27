
import numpy as np
import torch

from .. import setting
from . import activations
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
        if block_setting.nodes[0] == block_setting.nodes[-1] and \
                not self.create_linear_weight:
            self.linear_weight = activations.identity
        else:
            self.linear_weight = torch.nn.Linear(
                block_setting.nodes[0], block_setting.nodes[-1], bias=False)
        self.contraction = Contraction(setting.BlockSetting())
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
        h = self.contraction(x)
        linear_x = self.linear_weight(x)
        for linear, dropout_ratio, activation in zip(
                self.linears, self.dropout_ratios, self.activations):
            h = linear(h)
            h = torch.nn.functional.dropout(
                h, p=dropout_ratio, training=self.training)
            h = activation(h)
        return torch.einsum('i...f,if->i...f', linear_x, h)

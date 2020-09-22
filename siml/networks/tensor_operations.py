
import torch

from . import siml_module


class Contraction(siml_module.SimlModule):
    """Tensor contraction block."""

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        if self.block_setting.activations[0] != 'identity':
            raise ValueError(
                f"Cannot set nonlinear activation for: {self.block_setting}")
        self.dict_key = block_setting.optional.get('dict_key', None)
        return

    def forward(self, *xs, supports=None, original_shapes=None):
        """Calculate tensor contraction of rank n ( > m) and m tensors
        \\sum_{l_1, ..., l_m}
        A_{i,k_1,k_2,...,l_1,l_2,...,l_{m}} B_{i,l_1,l_2,...,l_m}
        """
        if len(xs) != 2:
            raise ValueError(f"2 inputs expected. Given: {len(xs)}")
        x = xs[0]
        y = xs[1]
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
        return torch.einsum(f"{string_x},{string_y}->{string_res}", x, y)

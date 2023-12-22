import numpy as np
import torch

from . import siml_module


class CrossProduct(siml_module.SimlModule):
    """Cross Product block."""

    @staticmethod
    def get_name():
        return 'cross_product'

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
        """Calculate cross product of tensors

        Parameters
        ----------
        xs: torch.Tensor
            [n_vertex, n_time_series, dim, n_feature]-shaped tensor. (dim = 3)
            OR [n_vertex, dim, n_feature]-shaped tensor
        """
        if len(xs) != 2:
            raise ValueError(f"2 inputs are expected. Given: {len(xs)}")

        x0 = xs[0]
        x1 = xs[1]
        if x0.shape[-2] != 3 or x1.shape[-2] != 3:
            raise ValueError(
                f"dimensions is not equal to 3."
                f"dim of x is {x0.shape[-2]}, dim of y is {x1.shape[-2]}"
            )
        if x0.shape[-1] != x1.shape[-1]:
            raise ValueError(
                "The number of x1 features is not equal to that of x0. "
                f"x0 shape: {x0.shape}, x1 shape: {x1.shape}"
            )

        return torch.stack([
            x0[..., 1, :] * x1[..., 2, :] - x0[..., 2, :] * x1[..., 1, :],
            x0[..., 2, :] * x1[..., 0, :] - x0[..., 0, :] * x1[..., 2, :],
            x0[..., 0, :] * x1[..., 1, :] - x0[..., 1, :] * x1[..., 0, :]
        ], axis=-2)

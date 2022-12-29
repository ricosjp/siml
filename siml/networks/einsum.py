
import numpy as np
import torch

from . import siml_module


class EinSum(siml_module.SimlModule):
    """EinSum block."""

    @staticmethod
    def get_name():
        return 'einsum'

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
        return np.max([
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

        if self.block_setting.input_names is None:
            raise ValueError(f"Put input_names for: {self.block_setting}")
        if 'equation' not in self.block_setting.optional:
            raise ValueError(
                f"Put optional.equation for: {self.block_setting}")
        self.equation = self.block_setting.optional.get('equation')

        return

    def forward(self, *xs, supports=None, original_shapes=None):
        """Calculate EinSum."""
        return torch.einsum(self.equation, *xs)

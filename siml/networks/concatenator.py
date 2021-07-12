
import torch

from . import siml_module


class Concatenator(siml_module.SimlModule):
    """Concatenation operation block."""

    @staticmethod
    def get_name():
        return 'concatenator'

    @staticmethod
    def accepts_multiple_inputs():
        return True

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def uses_support():
        return False

    @classmethod
    def _get_n_input_node(
            cls, block_setting, predecessors, dict_block_setting,
            input_length):
        return sum([
            dict_block_setting[predecessor].nodes[-1]
            for predecessor in predecessors])
        return input_length

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length):
        return input_node

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        return

    def forward(self, *xs, op=None, supports=None, original_shapes=None):
        return self.activation(torch.cat(xs, dim=-1))

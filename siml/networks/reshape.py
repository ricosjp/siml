
import torch

from . import siml_module


class Reshape(siml_module.SimlModule):
    """Reshape block."""

    @staticmethod
    def get_name():
        return 'reshape'

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return False

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length):
        return block_setting.optional['new_shape'][-1]

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        self.new_shape = block_setting.optional['new_shape']
        return

    def forward(self, x, supports=None, original_shapes=None):
        return torch.reshape(x, self.new_shape)

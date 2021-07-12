
import torch

from . import siml_module


class Array2Symmat(siml_module.SimlModule):
    """Convert array to symmetric matrix."""

    @staticmethod
    def get_name():
        return 'array2symmat'

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
    def _get_n_input_node(
            cls, block_setting, predecessors, dict_block_setting,
            input_length):
        return 6

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length):
        return 1

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        self.from_engineering = block_setting.optional.get(
            'from_engineering', True)
        self.indices_array2symmat = [0, 3, 5, 3, 1, 4, 5, 4, 2]
        return

    def forward(self, x, supports=None, original_shapes=None):
        n_feature = x.shape[1] // 6
        if x.shape[1] % 6 != 0:
            raise ValueError(f"Expected (n, 6*m) shape (given: {x.shape}")
        x = torch.stack([
            x[:, i_feature*6:(i_feature+1)*6]
            for i_feature in range(n_feature)])
        if self.from_engineering:
            x[:, :, 3:] = x[:, :, 3:] / 2
        y = torch.stack([
            torch.reshape(_x[:, self.indices_array2symmat], (-1, 3, 3))
            for _x in x], dim=-1)
        return y

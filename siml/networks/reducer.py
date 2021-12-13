
import numpy as np
import torch

from . import activations
from . import siml_module


class Reducer(siml_module.SimlModule):
    """Broadcastive operation block."""

    @staticmethod
    def get_name():
        return 'reducer'

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
        """Initialize the NN.

        Parameters
        -----------
        block_setting: siml.setting.BlockSetting
            BlockSetting object.
        """
        super().__init__(block_setting, no_parameter=True)

        if 'operator' in block_setting.optional:
            str_op = block_setting.optional['operator']
            if str_op == 'add':
                self.op = torch.add
            elif str_op == 'mul':
                self.op = torch.mul
            else:
                raise ValueError(f"Unknown operator for reducer: {str_op}")
        else:
            self.op = torch.add
            self.block_setting.optional['operator'] = 'add'
            print(f"optional.operator = add is set for: {block_setting}")

        self.split_keys = block_setting.optional.get('split_keys', None)

        return

    def forward(self, *xs, op=None, supports=None, original_shapes=None):
        if len(xs) == 1:
            raise ValueError(f"At least 2 inputs expected. Given: {len(xs)}")

        x = xs[0]
        for i_other, other in zip(range(1, len(xs)), xs[1:]):
            len_x = len(x.shape)
            len_other = len(other.shape)

            if x.shape[0] == other.shape[0]:
                if len_x == len_other:
                    x = self.op(x, other)
                elif len_x >= len_other:
                    axes = self._get_permute_axis(len_x, len_other)
                    x = self.op(x.permute(axes), other)
                    x = self._inverse_permute(x, axes)
                else:
                    if x.shape[0] == other.shape[0]:
                        axes = self._get_permute_axis(len_other, len_x)
                        x = self.op(x, other.permute(axes))
                        x = self._inverse_permute(x, axes)
                    else:
                        x = self._broadcast_batchsize(
                            self.op, x, other, original_shapes)

            else:
                if self.split_keys is None:
                    x = self._broadcast_batchsize(
                        self.op, x, other, original_shapes, None)
                else:
                    x = self._broadcast_batchsize(
                        self.op, x, other, original_shapes[self.split_keys[0]],
                        original_shapes[self.split_keys[i_other]])

        return self.activation(x)

    def _broadcast_batchsize(
            self, op, x, other, original_shapes, other_original_shapes):

        if other_original_shapes is None:
            if x.shape[0] >= other.shape[0]:
                split_data = activations.split(x, original_shapes)
                smaller = other
            elif x.shape[0] < other.shape[0]:
                split_data = activations.split(other, original_shapes)
                smaller = x
            else:
                raise ValueError('Shoud not reach here')
            return torch.cat([
                op(sd, smaller[i]) for i, sd in enumerate(split_data)])

        else:
            split_data = activations.split(x, original_shapes)
            other_split_data = activations.split(other, other_original_shapes)
            return torch.cat([
                op(sd, osd) for sd, osd in zip(split_data, other_split_data)])

    def _get_permute_axis(self, len_x, len_other):
        axes = list(range(len_other - 1, len_x - 1)) \
            + list(range(len_other - 1)) + [len_x - 1]
        return axes

    def _inverse_permute(self, x, axes):
        inverse_axes = np.argsort(axes)
        return x.permute(list(inverse_axes))

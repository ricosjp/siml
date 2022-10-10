import torch
from typing import Callable
from . import siml_module


class Projection(siml_module.SimlModule):

    @staticmethod
    def get_name():
        return 'projection'

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
    def _get_n_input_node(cls,
                          block_setting,
                          predecessors,
                          dict_block_setting,
                          input_length,
                          **kwargs):
        return dict_block_setting[predecessors[1]].nodes[-1]

    @classmethod
    def _get_n_output_node(cls,
                           input_node,
                           block_setting,
                           predecessors,
                           dict_block_setting,
                           output_length,
                           **kwargs):
        return input_node

    def __init__(self,
                 block_setting):
        super().__init__(block_setting,
                         no_parameter=True,
                         create_activations=False)

        self.op = self._select_operator(block_setting)
        self.time_series_input = self._is_time_series_input(block_setting)

    def _select_operator(self, block_setting) -> Callable:
        if 'operator' not in block_setting.optional:
            # default is mean function
            return torch.mean

        str_op = block_setting.optional['operator']
        if str_op == 'mean':
            return torch.mean
        elif str_op == 'sum':
            return torch.sum
        else:
            raise ValueError(f"Unknown operator for projection: {str_op}")

    def _is_time_series_input(self, block_setting) -> bool:
        if 'time_series_input' not in block_setting.optional:
            return False

        return block_setting.optional['time_series_input']

    def forward(self,
                *xs,
                supports=None,
                original_shapes=None) -> torch.Tensor:
        if len(xs) != 2:
            raise ValueError(
                f"Input tensors should be x and projection. ({len(xs)} given)"
            )

        x = xs[0]
        projection = torch.flatten(xs[1])
        max_n = self._get_max_integer(projection)
        if max_n == 0:
            ValueError('projection marker is not found.')

        for flag in range(1, max_n + 1):
            filter_from = (projection == int(-1) * flag)
            filter_to = (projection == flag)

            if not self.time_series_input:
                # stedy state tensor
                v = self.op(x[filter_from, ...], axis=0)
                x[filter_to, ...] = v
            else:
                # time series tensor
                v = self.op(x[:, filter_from, ...], axis=1)
                n_to = torch.sum(filter_to).item()
                val_size = v.shape[-1]
                x[:, filter_to, ...] = \
                    v.repeat(1, n_to).view(-1, n_to, val_size)

        return x

    def _get_max_integer(self, tensor: torch.Tensor) -> int:
        fileter_not_nan = ~torch.isnan(tensor)
        not_nan_tensor = tensor[..., fileter_not_nan]
        int_val = int(torch.max(not_nan_tensor).cpu().numpy().item())
        return int_val

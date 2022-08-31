import torch
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
        return dict_block_setting[predecessors[0]].nodes[-1]

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

        if 'operator' in block_setting.optional:
            str_op = block_setting.optional['operator']
            if str_op == 'mean':
                self.op = torch.mean
            elif str_op == 'sum':
                self.op = torch.sum
            else:
                raise ValueError(f"Unknown operator for projection: {str_op}")
        else:
            self.op = torch.mean

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
            filter_to = (projection == int(-1) * flag)
            fillter_from = (projection == flag)
            v = self.op(x[fillter_from, ...], axis=0)
            x[filter_to, ...] = v

        return x

    def _get_max_integer(self, tensor: torch.Tensor) -> int:
        fileter_not_nan = ~torch.isnan(tensor)
        not_nan_tensor = tensor[..., fileter_not_nan]
        int_val = int(torch.max(not_nan_tensor).cpu().numpy().item())
        return int_val

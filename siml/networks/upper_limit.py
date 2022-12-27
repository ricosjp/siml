from . import siml_module
import torch


class UpperLimit(siml_module.SimlModule):
    @staticmethod
    def get_name():
        return 'upper_limit'

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def accepts_multiple_inputs():
        return True

    @staticmethod
    def uses_support():
        return False

    def __init__(self, block_setting):
        super().__init__(block_setting,
                         create_activations=False,
                         no_parameter=True)

    def forward(self,
                *xs,
                supports=None,
                original_shapes=None) -> torch.Tensor:
        x = xs[0]
        max_values = xs[1]

        if x.shape != max_values.shape:
            raise ValueError("max_value has the same shape as input."
                             f"max_value is {max_values.shape} given, "
                             f"inputs {x.shape} given")

        flags = x > max_values
        y = x * ~flags + max_values * flags
        return y

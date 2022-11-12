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
        max_value: torch.Tensor = xs[1]

        if max_value.numel() != 1:
            raise ValueError("max_value has only one element."
                             f"{max_value.numel()} given")

        x[x > max_value] = max_value
        return x

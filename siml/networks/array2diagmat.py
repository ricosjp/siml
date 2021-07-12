
import torch

from . import siml_module


class Array2Diagmat(siml_module.SimlModule):
    """Convert array to diagonal matrix. [n, f] -> [n, 3, 3, f]"""

    @staticmethod
    def get_name():
        return 'array2diagmat'

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return False

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        return

    def forward(self, x, supports=None, original_shapes=None):
        eye = torch.eye(3, device=x.device)
        y = torch.einsum('ik,mn->imnk', x, eye)
        return y

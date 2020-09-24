
import torch

from . import siml_module


class Array2Diagmat(siml_module.SimlModule):
    """Convert array to diagonal matrix. [n, f] -> [n, 3, 3, f]"""

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        return

    def forward(self, x, supports=None, original_shapes=None):
        eye = torch.eye(3, device=x.device)
        y = torch.einsum('ik,mn->imnk', x, eye)
        return y

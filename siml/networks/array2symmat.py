
import torch

from . import siml_module


class Array2Symmat(siml_module.SimlModule):
    """Convert array to symmetric matrix."""

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

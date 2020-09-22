
import torch

from . import siml_module


class Symmat2Array(siml_module.SimlModule):
    """Convert symmetric matrix to array."""

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        self.to_engineering = block_setting.optional.get(
            'to_engineering', False)
        self.indices_symmat2array = [0, 4, 8, 1, 5, 2]
        return

    def forward(self, x, supports=None, original_shapes=None):
        n_feature = x.shape[-1]
        y = torch.reshape(x, (-1, 9, n_feature))[
            :, self.indices_symmat2array]
        if self.to_engineering:
            y[:, 3:, :] = y[:, 3:, :] * 2

        return torch.cat(
            [y[..., i_feature] for i_feature in range(n_feature)], dim=-1)

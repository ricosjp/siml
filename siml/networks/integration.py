
import einops
import torch

from . import header


class Integration(torch.nn.Module):
    """Integration block."""

    def __init__(self, block_setting):
        """Initialize the NN.

        Parameters
        -----------
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
        """
        super().__init__()
        if len(block_setting.activations) != 1:
            raise ValueError(
                f"Invalid activation length: {len(block_setting.activations)} "
                f"for {block_setting}")
        self.activation = header.DICT_ACTIVATIONS[
            block_setting.activations[0]]
        if 'dummy_index' in block_setting.optional:
            self.dummy_index = block_setting.optional['dummy_index']
        else:
            self.dummy_index = 0
        return

    def _diff(self, x):
        shape = x.shape
        x = einops.rearrange(
            x, 'time batch element feature -> (batch element) feature time')
        x = torch.nn.functional.pad(
            x, (1, 0), mode='reflect')
        x = einops.rearrange(
            x, '(batch element) feature time -> time batch element feature',
            batch=shape[1], element=shape[2])
        x[0] = -x[0]
        return x[1:] - x[:-1]

    def forward(self, x, supports=None):
        f = torch.cat(
            [x[..., :self.dummy_index], x[..., self.dummy_index + 1:]], dim=-1)
        t = x[..., [self.dummy_index]]
        dt = self._diff(t)
        f_dt = f * dt
        integrated = torch.cumsum(f_dt, dim=0)
        return self.activation(integrated)

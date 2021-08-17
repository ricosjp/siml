
import einops
import torch

from . import siml_module


class Integration(siml_module.SimlModule):
    """Integration block."""

    @staticmethod
    def get_name():
        return 'integration'

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return False

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length):
        return input_node - 1

    def __init__(self, block_setting):
        """Initialize the NN.

        Parameters
        -----------
        block_setting: siml.setting.BlockSetting
            BlockSetting object.
        """
        super().__init__(block_setting, no_parameter=True)
        if 'dummy_index' in block_setting.optional:
            self.dummy_index = block_setting.optional['dummy_index']
        else:
            self.dummy_index = 0
        return

    def _pad(self, x):
        x = einops.rearrange(
            x, 'time element feature -> element feature time')
        x = torch.nn.functional.pad(
            x, (1, 0), mode='replicate')
        x = einops.rearrange(
            x, 'element feature time -> time element feature')
        return x

    def _diff(self, x):
        x = self._pad(x)
        return x[1:] - x[:-1]

    def _integrate(self, t, f):
        dt = self._diff(t)
        f = self._pad(f)
        return torch.cumsum((f[1:] + f[:-1]) * dt * .5, dim=0)

    def forward(self, x, supports=None, original_shapes=None):
        t = x[..., [self.dummy_index]]
        f = torch.cat(
            [x[..., :self.dummy_index], x[..., self.dummy_index + 1:]], dim=-1)
        integrated = self._integrate(t, f)
        return self.activation(integrated)

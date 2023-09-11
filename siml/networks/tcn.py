
import einops
import numpy as np
import torch

from . import activations
from . import siml_module


class TCN(siml_module.SimlModule):
    """Temporal Convolutional Networks (TCN) https://arxiv.org/abs/1803.01271 .
    """

    @staticmethod
    def get_name():
        return 'tcn'

    @staticmethod
    def is_trainable():
        return True

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return False

    def __init__(self, block_setting):
        """Initialize the NN.

        Parameters
        -----------
        block_setting: siml.setting.BlockSetting
            BlockSetting object.
        """

        super().__init__(block_setting, create_linears=False)
        nodes = block_setting.nodes
        kernel_sizes = block_setting.kernel_sizes
        if 'dilations' in block_setting.optional:
            dilations = block_setting.optional['dilations']
            if len(dilations) != len(kernel_sizes):
                raise ValueError(
                    f"len(dilations) should be {len(kernel_sizes)} but"
                    f"{len(dilations)} for {block_setting}")
        else:
            dilations = [1] * len(kernel_sizes)
        self.padding_lengths = (
            np.array(kernel_sizes) - 1) * np.array(dilations)

        if 'padding_modes' in block_setting.optional:
            self.padding_modes = block_setting.optional['padding_modes']
            if len(self.padding_modes) != len(kernel_sizes):
                raise ValueError(
                    f"len(padding_modes) should be {len(kernel_sizes)} but"
                    f"{len(self.padding_modes)} for {block_setting}")
        else:
            self.padding_modes = ['replicate'] * len(kernel_sizes)

        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(
                n1, n2, k, dilation=dilation, bias=self.block_setting.bias)
            for n1, n2, k, dilation
            in zip(nodes[:-1], nodes[1:], kernel_sizes, dilations)])

        return

    def _pad_front(self, x, padding_mode, padding_length):
        return torch.nn.functional.pad(
            x, (padding_length, 0), mode=padding_mode)

    def _forward_core(self, x, supports=None, original_shapes=None):
        """Execute the NN's forward computation.

        Parameters
        -----------
        x: numpy.ndarray or cupy.ndarray
            Input of the NN.
        supports: list[chainer.util.CooMatrix]
            List of support inputs.

        Returns
        --------
        y: numpy.ndarray of cupy.ndarray
            Output of the NN.
        """
        original_shape = x.shape
        if len(original_shape) == 3:
            kwargs = {}
            forward_pattern = 'time element feature -> element feature time'
            backward_pattern = 'element feature time -> time element feature'
        else:
            kwargs = {
                f"shape{i}": s for i, s in enumerate(original_shape[2:-1])}
            dims = ' '.join(
                f"shape{i}" for i, s in enumerate(original_shape[2:-1]))
            forward_pattern = f"time element {dims} feature -> " \
                + f"(element {dims}) feature time"
            backward_pattern = f"(element {dims}) feature time -> " \
                + f"time element {dims} feature"
        h = einops.rearrange(x, forward_pattern, **kwargs)

        for conv, padding_mode, padding_length, dropout_ratio, activation \
                in zip(
                    self.convs, self.padding_modes, self.padding_lengths,
                    self.dropout_ratios, self.activations):
            # Pad in the time direction
            h = self._pad_front(h, padding_mode, padding_length)
            h = conv(h)
            h = torch.nn.functional.dropout(
                h, p=dropout_ratio, training=self.training)
            h = activation(h)

        h = einops.rearrange(h, backward_pattern, **kwargs)
        return h


class EquivariantTCN(siml_module.SimlModule):
    """E(n)-equivariant TCN layer."""

    @staticmethod
    def get_name():
        return 'equivariant_tcn'

    @staticmethod
    def is_trainable():
        return True

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return False

    def __init__(self, block_setting):
        """Initialize the NN.

        Parameters
        -----------
        block_setting: siml.setting.BlockSetting
            BlockSetting object.
        """
        super().__init__(block_setting, create_linears=False)
        self.tcn = TCN(block_setting)

        if self.block_setting.nodes[0] == self.block_setting.nodes[-1]:
            self.linear = activations.identity
        else:
            self.linear = torch.nn.Linear(
                self.block_setting.nodes[0],
                self.block_setting.nodes[-1],
                bias=False)
        return

    def _forward_core(self, x, supports=None, original_shapes=None):
        """Execute the NN's forward computation.

        Parameters
        -----------
        x: numpy.ndarray or cupy.ndarray
            Input of the NN.
        supports: list[chainer.util.CooMatrix]
            List of support inputs.

        Returns
        --------
        y: numpy.ndarray of cupy.ndarray
            Output of the NN.
        """
        ts_norm = torch.einsum('tn...f,tn...f->tnf', x, x)
        ts_norm_h = self.tcn(ts_norm)
        h = self.linear(x)
        return torch.einsum('tn...f,tnf->tn...f', h, ts_norm_h)


import einops
import numpy as np
import torch

from . import header


class TCN(torch.nn.Module):
    """Temporal Convolutional Networks (TCN) https://arxiv.org/abs/1803.01271 .
    """

    def __init__(self, block_setting):
        """Initialize the NN.

        Parameters
        -----------
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
        """

        super().__init__()
        self.residual = block_setting.residual
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
        # self.padding_lengths = np.array(kernel_sizes) - 1

        if 'padding_modes' in block_setting.optional:
            self.padding_modes = block_setting.optional['padding_modes']
            if len(self.padding_modes) != len(kernel_sizes):
                raise ValueError(
                    f"len(padding_modes) should be {len(kernel_sizes)} but"
                    f"{len(self.padding_modes)} for {block_setting}")
        else:
            self.padding_modes = ['replicate'] * len(kernel_sizes)

        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(n1, n2, k, dilation=dilation)
            for n1, n2, k, dilation
            in zip(nodes[:-1], nodes[1:], kernel_sizes, dilations)])

        if self.residual:
            activations = [
                header.DICT_ACTIVATIONS[activation]
                for activation in block_setting.activations]
            self.activations = activations[:-1] \
                + [header.DICT_ACTIVATIONS['identity']] \
                + [activations[-1]]
            if nodes[0] == nodes[-1]:
                self.shortcut = header.identity
            else:
                self.shortcut = torch.nn.Linear(nodes[0], nodes[-1])
        else:
            self.activations = [
                header.DICT_ACTIVATIONS[activation]
                for activation in block_setting.activations]

        self.dropout_ratios = [
            dropout_ratio for dropout_ratio in block_setting.dropouts]

        return

    def _pad_front(self, x, padding_mode, padding_length):
        return torch.nn.functional.pad(
            x, (padding_length, 0), mode=padding_mode)

    def forward(self, x, supports=None):
        """Execute the NN's forward computation.

        Parameters
        -----------
            x: numpy.ndarray or cupy.ndarray
                Input of the NN.
            supports: List[chainer.util.CooMatrix]
                List of support inputs.
        Returns
        --------
            y: numpy.ndarray of cupy.ndarray
                Output of the NN.
        """
        shape = x.shape
        h = einops.rearrange(
            x, 'time batch element feature -> (batch element) feature time')

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

        h = einops.rearrange(
            h, '(batch element) feature time -> time batch element feature',
            batch=shape[1], element=shape[2])

        if self.residual:
            h = self.activations[-1](h + self.shortcut(x))
            return h
        else:
            return h

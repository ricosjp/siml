
import torch

from . import header


class LSTM(torch.nn.Module):
    """LSTM layer."""

    def __init__(self, block_setting, residual=False):
        """Initialize the NN.

        Parameters
        -----------
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
            residual: bool
                If True, use residual network.
        """

        super().__init__()
        nodes = block_setting.nodes
        self.residual = residual
        self.lstm_layers = torch.nn.ModuleList([
            torch.nn.LSTM(n1, n2, dropout=do)
            for n1, n2, do
            in zip(nodes[:-1], nodes[1:], block_setting.dropouts)])
        if self.residual:
            if nodes[0] == nodes[1]:
                self.linear = header.identity
            else:
                self.linear = torch.nn.Linear(nodes[0], nodes[-1])

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
        h = x.view(
            shape[0], shape[1] * shape[2], -1)
        hidden = None
        for lstm_layer in self.lstm_layers:
            h, hidden = lstm_layer(h, hidden)

        if self.residual:
            return h.view(shape[0], shape[1], shape[2], h.shape[-1]) \
                + self.linear(x)
        else:
            return h.view(shape[0], shape[1], shape[2], h.shape[-1])


class ResLSTM(LSTM):

    def __init__(self, block_setting):
        super().__init__(block_setting, residual=True)

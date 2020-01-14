
import torch


class LSTM(torch.nn.Module):
    """LSTM layer."""

    def __init__(self, block_setting):
        """Initialize the NN.

        Parameters
        -----------
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
        """

        super().__init__()
        nodes = block_setting.nodes
        self.lstm_layers = torch.nn.ModuleList([
            torch.nn.LSTM(n1, n2, dropout=do)
            for n1, n2, do
            in zip(nodes[:-1], nodes[1:], block_setting.dropouts)])
        self.input_selection = block_setting.input_selection

    def __call__(self, x, supports=None):
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
        h = x[:, :, :, self.input_selection].view(shape[0], shape[1], -1)
        for lstm_layer in self.lstm_layers:
            h, _ = lstm_layer(h)
        return h.view(shape[0], shape[1], shape[2], h.shape[-1])

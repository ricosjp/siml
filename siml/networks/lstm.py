
import torch

from . import siml_module


class LSTM(siml_module.SimlModule):
    """LSTM layer."""

    @staticmethod
    def get_name():
        return 'lstm'

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
        residual: bool
            If True, use residual network.
        """

        super().__init__(block_setting, create_linears=False)
        nodes = block_setting.nodes
        self.residual = block_setting.residual
        self.lstm_layers = torch.nn.ModuleList([
            torch.nn.LSTM(n1, n2, dropout=do)
            for n1, n2, do
            in zip(nodes[:-1], nodes[1:], block_setting.dropouts)])
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
        h = x
        hidden = None
        for lstm_layer in self.lstm_layers:
            h, hidden = lstm_layer(h, hidden)
        return h

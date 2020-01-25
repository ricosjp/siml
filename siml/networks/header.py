
import torch

from . import adjustable_mlp
from .. import setting


def identity(x):
    return x


def max_pool(x):
    return torch.max(x, dim=-2, keepdim=True)[0]


def mean(x):
    return torch.mean(x, dim=-2, keepdim=True)


DICT_ACTIVATIONS = {
    'identity': identity,
    'relu': torch.relu,
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
    'max_pool': max_pool,
    'max': max_pool,
    'mean': mean,
}


class AbstractMLP(torch.nn.Module):

    def __init__(self, block_setting, last_identity=False):
        """Initialize the NN.

        Parameters
        -----------
        block_setting: siml.setting.BlockSetting
            BlockSetting object.
        last_identity: bool
            If True, set the last activation identity whatever the
            block_setting (default: False).
        """
        super().__init__()

        nodes = block_setting.nodes
        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(n1, n2) for n1, n2 in zip(nodes[:-1], nodes[1:])])
        self.activations = [
            DICT_ACTIVATIONS[activation]
            for activation in block_setting.activations]
        if last_identity:
            self.activations[-1] = DICT_ACTIVATIONS['identity']
        self.dropout_ratios = [
            dropout_ratio for dropout_ratio in block_setting.dropouts]
        self.input_selection = block_setting.input_selection

    def forward(self, x, support=None):
        raise NotImplementedError


class AbstractGCN(torch.nn.Module):

    def __init__(
            self, block_setting,
            *, create_subchain=True, adjustable_subchain=False):
        """Initialize the NN.

        Parameters
        -----------
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
            create_subchain: bool, optional [True]
                If True, create subchain to be trained.
            adjustable_subchain: bool, optional [False]
                If True, create subchain as a stack of AdjustableMLP layers
                instead of that of toch.nn.Linear layers.
        """

        super().__init__()

        nodes = block_setting.nodes
        if adjustable_subchain:
            block_settings = [
                setting.BlockSetting(
                    nodes=[n1, n2], activations=['identity'],
                    dropouts=[dropout])
                for n1, n2, dropout
                in zip(nodes[:-1], nodes[1:], block_setting.dropouts)]
            self.subchains = torch.nn.ModuleList([
                adjustable_mlp.AdjustableMLP(bs)
                for bs in block_settings
            ])
        else:
            self.subchains = torch.nn.ModuleList([
                torch.nn.Linear(n1, n2)
                for n1, n2 in zip(nodes[:-1], nodes[1:])])

        self.activations = [
            DICT_ACTIVATIONS[activation]
            for activation in block_setting.activations]
        self.dropout_ratios = [
            dropout_ratio for dropout_ratio in block_setting.dropouts]
        self.support_input_index = block_setting.support_input_index
        self.input_selection = block_setting.input_selection

    def forward(self, x, supports):
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
        if len(x.shape) == 3:
            hs = torch.stack([
                self._forward_single(
                    x_[:, self.input_selection],
                    supports_[self.support_input_index])
                for x_, supports_ in zip(x, supports)])
        else:
            hs = torch.stack([
                torch.stack([
                    self._forward_single(
                        x__[:, self.input_selection],
                        supports_[self.support_input_index])
                    for x__, supports_ in zip(x_, supports)])
                for x_ in x])
        return hs

    def _forward_single(self, x, support):
        raise NotImplementedError


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

        self.multiple_networks = block_setting.optional.get(
            'multiple_networks', True)
        if create_subchain:
            self.subchains, self.subchain_indices = self._create_subchains(
                block_setting, adjustable_subchain)

        self.activations = [
            DICT_ACTIVATIONS[activation]
            for activation in block_setting.activations]
        self.dropout_ratios = [
            dropout_ratio for dropout_ratio in block_setting.dropouts]

    def _create_subchains(
            self, block_setting, adjustable_subchain=False,
            twice_input_nodes=False, square_weight=False, start_index=0):
        if self.multiple_networks:
            subchains = torch.nn.ModuleList([
                self._create_subsubchain(
                    block_setting,
                    adjustable_subchain=adjustable_subchain,
                    twice_input_nodes=twice_input_nodes,
                    square_weight=square_weight, start_index=start_index)
                for _ in block_setting.support_input_indices])
            subchain_indices = range(
                len(block_setting.support_input_indices))
        else:
            subchains = torch.nn.ModuleList([
                self._create_subsubchain(
                    block_setting,
                    adjustable_subchain=adjustable_subchain,
                    twice_input_nodes=twice_input_nodes,
                    square_weight=square_weight, start_index=start_index)])
            subchain_indices = [0] * len(
                block_setting.support_input_indices)

        return subchains, subchain_indices

    def _create_subsubchain(
            self, block_setting, adjustable_subchain=False,
            twice_input_nodes=False, square_weight=False, start_index=0):
        nodes = block_setting.nodes[start_index:]
        if twice_input_nodes:
            factor = 2
        else:
            factor = 1

        if square_weight:
            node_tuples = [(n, n) for n in nodes]
        else:
            node_tuples = [
                (n1 * factor, n2) for n1, n2 in zip(nodes[:-1], nodes[1:])]

        if adjustable_subchain:
            block_settings = [
                setting.BlockSetting(
                    nodes=node_tuple, activations=['identity'],
                    dropouts=[dropout])
                for node_tuple, dropout
                in zip(node_tuples, block_setting.dropouts)]
            return torch.nn.ModuleList([
                adjustable_mlp.AdjustableMLP(bs)
                for bs in block_settings
            ])
        else:
            return torch.nn.ModuleList([
                torch.nn.Linear(*node_tuple) for node_tuple in node_tuples])

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
                self._forward_single(x_, supports_)
                for x_, supports_ in zip(x, supports)])
        else:
            hs = torch.stack([
                torch.stack([
                    self._forward_single(x__, supports_)
                    for x__, supports_ in zip(x_, supports)])
                for x_ in x])
        return hs

    def _forward_single(self, x, supports):
        try:
            return torch.sum(torch.stack([
                self._forward_single_core(x, self.subchain_indices[i], support)
                for i, support in enumerate(supports)]), dim=0)
        except IndexError:
            raise ValueError(self.subchain_indices, len(supports))

    def _forward_single_core(self, x, subchain_index, support):
        raise NotImplementedError

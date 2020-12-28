
import numpy as np
import torch

from . import abstract_pyg_gcn
from .gcnii_pytorch_geometric import GCN2Conv


class GCNII(abstract_pyg_gcn.AbstractPyGGCN):
    """GCNII based on https://arxiv.org/abs/2007.02133 ."""

    def __init__(self, block_setting):
        super().__init__(
            block_setting, create_subchain=False, residual=False)

        self.alpha = block_setting.optional.get('alpha', .5)
        # lambda in the paper
        self.theta = block_setting.optional.get('theta', 1.)
        self.shared_weights = block_setting.optional.get(
            'shared_weights', True)
        self.add_self_loop = block_setting.optional.get(
            'add_self_loop', False)  # NB: FEM adjacency has self loop
        if not np.all(np.array(block_setting.nodes) == block_setting.nodes[0]):
            raise ValueError(
                f"The number of node should be the same for: {block_setting}")
        self.channels = block_setting.nodes[0]

        self.gcniis = torch.nn.ModuleList([
            self._generate_gcniis() for _ in self.subchain_indices])
        return

    def _generate_gcniis(self):
        return torch.nn.ModuleList([
            GCN2Conv(
                channels=self.channels, alpha=self.alpha, theta=self.theta,
                layer=i_layer+1, shared_weights=self.shared_weights)
            for i_layer in range(len(self.block_setting.nodes) - 1)])

    def _forward_single_core(self, x, subchain_index, support):
        edge_index = support._indices()
        h = x
        for gcnii, dropout_ratio, activation in zip(
                self.gcniis[subchain_index], self.dropout_ratios,
                self.activations):
            h = gcnii(h, x, edge_index)
            h = torch.nn.functional.dropout(
                h, p=dropout_ratio, training=self.training)
            h = activation(h)
        return h

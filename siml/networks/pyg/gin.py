
import torch
import torch_geometric

from . import abstract_pyg_gcn
from .. import identity
from .. import mlp
from ... import setting


class GIN(abstract_pyg_gcn.AbstractPyGGCN):
    """Graph Isomorphism Network based on https://arxiv.org/abs/1810.00826 ."""

    def __init__(self, block_setting):
        super().__init__(
            block_setting, create_subchain=True, residual=False)

        self.epsilon = block_setting.optional.get('epsilon', 0.)
        self.train_epsilon = block_setting.optional.get('train_epsilon', False)

        self.gins = torch.nn.ModuleList([
            torch_geometric.nn.GINConv(
                identity.Identity(setting.BlockSetting()),
                eps=self.epsilon, train_eps=self.train_epsilon)
            for _ in self.subchains])
        block_setting_for_mlp = setting.BlockSetting(
            type='mlp', nodes=self.block_setting.nodes,
            activations=self.block_setting.activations,
            dropouts=self.block_setting.dropouts)

        self.mlps = torch.nn.ModuleList([
            mlp.MLP(block_setting_for_mlp) for _ in self.subchains])
        return

    def _forward_single_core(self, x, subchain_index, support):
        edge_index = self._remove_self_loop_if_exists(support)
        return self.mlps[subchain_index](
            self.gins[subchain_index](x, edge_index))

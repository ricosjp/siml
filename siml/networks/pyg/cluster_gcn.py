
import torch
import torch_geometric

from . import abstract_pyg_gcn


class ClusterGCN(abstract_pyg_gcn.AbstractPyGGCN):
    """Cluster-GCN based on https://arxiv.org/abs/1905.07953 ."""

    def __init__(self, block_setting):
        super().__init__(
            block_setting, create_subchain=False, residual=False)

        self.diag_lambda = block_setting.optional.get('diag_lambda', 0.)
        self.add_self_loops = block_setting.optional.get(
            'add_self_loops', False)  # NB: FEM adjacency has self loop

        self.cluster_gcns = torch.nn.ModuleList([
            self._generate_cluster_gcns() for _ in self.subchain_indices])
        return

    def _generate_cluster_gcns(self):
        nodes = self.block_setting.nodes
        return torch.nn.ModuleList([
            torch_geometric.nn.ClusterGCNConv(
                n1, n2, diag_lambda=self.diag_lambda,
                add_self_loops=self.add_self_loops,
                bias=self.block_setting.bias)
            for n1, n2 in zip(nodes[:-1], nodes[1:])])

    def _forward_single_core(self, x, subchain_index, support):
        edge_index = support._indices()
        size = support.size()
        h = x
        for cluster_gcn, dropout_ratio, activation in zip(
                self.cluster_gcns[subchain_index], self.dropout_ratios,
                self.activations):
            h = cluster_gcn(h, edge_index, size=size)
            h = torch.nn.functional.dropout(
                h, p=dropout_ratio, training=self.training)
            h = activation(h)
        return h

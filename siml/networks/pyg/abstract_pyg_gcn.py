import torch_geometric

from .. import abstract_gcn


class AbstractPyGGCN(abstract_gcn.AbstractGCN):

    def _remove_self_loop_if_exists(self, support):
        edge_index = support._indices()
        if self.keep_self_loop:
            return edge_index

        if torch_geometric.utils.contains_self_loops(edge_index):
            return_edge_index, _ = torch_geometric.utils.remove_self_loops(
                edge_index)
        else:
            return_edge_index = support._indices()
        return return_edge_index

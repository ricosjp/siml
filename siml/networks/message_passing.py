
import numpy as np
import scipy.sparse as sp
import torch

from . import abstract_gcn
from . import spmm
from .. import setting


class MessagePassing(abstract_gcn.AbstractGCN):
    """Message passing layer based on https://arxiv.org/abs/1704.01212.
    """

    @staticmethod
    def get_name():
        return 'message_passing'

    def __init__(self, block_setting):
        """Initialize the NN.

        Parameters
        ----------
        block_setting: siml.setting.BlockSetting
            BlockSetting object.
        """
        super().__init__(block_setting, create_subchain=False)
        self.concat = block_setting.optional.get(
            'concat', True)
        self.edge_parameters, self.subchain_indices = self._create_subchains(
            block_setting, block_setting.nodes, twice_input_nodes=self.concat)
        self.node_parameters, _ = self._create_subchains(
            block_setting, block_setting.nodes, square_weight=True,
            start_index=1)
        self.spmm = spmm.SpMM(setting.BlockSetting(optional={
            'mode': 'mean', 'transpose': True}))
        return

    def make_reduce_matrix(self, nadj, *, mean=False):
        col = nadj._indices()[1].cpu().numpy()
        data = np.ones(len(col), dtype=np.float32)
        row = np.arange(len(col))
        shape = torch.Size((len(row), nadj.shape[0]))

        if mean:
            rm = sp.coo_matrix((data, (row, col)))
            degrees = np.array(rm.sum(0))
            normalized_rm = rm.multiply(1. / degrees)

            reduce_matrix = torch.sparse_coo_tensor(
                torch.LongTensor(np.array([row, col])),
                torch.FloatTensor(normalized_rm.data), shape)
        else:
            reduce_matrix = torch.sparse_coo_tensor(
                torch.LongTensor(np.array([row, col])),
                torch.FloatTensor(data), shape)

        return reduce_matrix.to(nadj.device)

    def _forward_single_core(self, x, subchain_index, support):
        """Execute the NN's forward computation.

        Parameters
        ----------
        x: numpy.ndarray or cupy.ndarray
            Input of the NN.
        support: chainer.util.CooMatrix
            Normalized adjacency matrix.

        Returns
        --------
        y: numpy.ndarray of cupy.ndarray
            Output of the NN.
        """
        h_node = x
        reduce_matrix = self.make_reduce_matrix(support)
        row = support._indices()[0].cpu().numpy()
        col = support._indices()[1].cpu().numpy()
        for i in range(len(self.edge_parameters[subchain_index])):
            if self.concat:
                merged = torch.cat(
                    [h_node[col], h_node[row]], dim=-1)
            else:
                merged = h_node[col] - h_node[row]

            edge_emb = self.activations[i](
                self.edge_parameters[subchain_index][i](merged))

            h_edge = self.spmm(edge_emb, supports=[reduce_matrix])
            h_node = self.activations[i](
                self.node_parameters[subchain_index][i](h_edge))

        return h_node

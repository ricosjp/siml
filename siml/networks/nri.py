import chainer as ch
from chainer.backends import cuda
import numpy as np
import scipy.sparse as sp

from . import header
from .. import util


class NRI(header.AbstractGCN):
    """Neural Relational Inference layer based on
    https://arxiv.org/pdf/1802.04687.pdf .
    """

    def __init__(self, block_setting):
        """Initialize the NN.

        Args:
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
        """
        super().__init__(block_setting, create_subchain=False)
        self.concat = True
        with self.init_scope():
            if self.concat:
                self.edge_parameters = ch.ChainList(*[
                    ch.links.Linear(n1 * 2, n2) for n1, n2
                    in zip(block_setting.nodes[:-1], block_setting.nodes[1:])])
            else:
                self.edge_parameters = ch.ChainList(*[
                    ch.links.Linear(n1, n2) for n1, n2
                    in zip(block_setting.nodes[:-1], block_setting.nodes[1:])])
            self.node_parameters = ch.ChainList(*[
                ch.links.Linear(n, n) for n in block_setting.nodes[1:]])

    def make_reduce_matrix(self, nadj, *, mean=False):
        data = np.ones(len(nadj.col), dtype=np.float32)
        row = np.arange(len(nadj.col))
        if ch.cuda.available:
            col = ch.cuda.cupy.asnumpy(nadj.col)
        else:
            col = nadj.col
        shape = (len(row), nadj.shape[0])
        if mean:
            rm = sp.coo_matrix((data, (row, col)))
            degrees = np.array(rm.sum(0))
            normalized_rm = rm.multiply(1. / degrees)

            if hasattr(nadj.data.data, 'device'):
                reduce_matrix = util.convert_sparse_to_chainer(
                    normalized_rm, device=nadj.data.data.device.id)
            else:
                reduce_matrix = util.convert_sparse_to_chainer(
                    normalized_rm)
        else:
            reduce_matrix = ch.utils.CooMatrix(data, row, col, shape)

        return reduce_matrix

    def __call__(self, x, supports):
        """Execute the NN's forward computation.

        Args:
            x: numpy.ndarray or cupy.ndarray
                Input of the NN.
            supports: List[chainer.util.CooMatrix]
                List of support inputs.
        Returns:
            y: numpy.ndarray of cupy.ndarray
                Output of the NN.
        """
        hs = ch.functions.stack([
            self._call_single(
                x_[:, self.input_selection],
                supports_[self.support_input_index])
            for x_, supports_ in zip(x, supports)])
        return hs

    def _call_single(self, x, support):
        """Execute the NN's forward computation.

        Args:
            x: numpy.ndarray or cupy.ndarray
                Input of the NN.
            support: chainer.util.CooMatrix
                Normalized adjacency matrix.
        Returns:
            y: numpy.ndarray of cupy.ndarray
                Output of the NN.
        """

        # h_node = x
        # # 4 loop
        # reduce_matrix = self.make_reduce_matrix(support, mean=True)
        # for i in range(len(self.edge_parameters)):
        #     merged = ch.functions.concat([h_node[support.col],
        #                                   h_node[support.row]],
        #                                  axis=1)
        #     print(merged.shape, h_node.shape)
        #     edge_emb = ch.functions.relu(
        #         self.edge_parameters[i](merged))
        #     h_edge = ch.functions.sparse_matmul(
        #         reduce_matrix, edge_emb, transa=True)
        #     if self.last_layer and i == len(self.edge_parameters) - 1:
        #         # print('identity')
        #         h_node = self.node_parameters[i](h_edge)
        #     else:
        #         # print('relu')
        #         h_node = ch.functions.relu(self.node_parameters[i](h_edge))

        h_node = x
        # 4 loop
        reduce_matrix = self.make_reduce_matrix(support, mean=True)
        for i in range(len(self.edge_parameters)):
            xp = cuda.get_array_module(x)
            if hasattr(support.data.data, 'device'):
                with xp.cuda.Device(support.data.data.device):
                    if self.concat:
                        merged = ch.functions.concat(
                          [h_node[support.col], h_node[support.row]], axis=1)
                    else:
                        merged = h_node[support.col] - h_node[support.row]
            else:
                if self.concat:
                    merged = ch.functions.concat(
                      [h_node[support.col], h_node[support.row]], axis=1)
                else:
                    merged = h_node[support.col] - h_node[support.row]
            print(self.edge_parameters[i].W.shape, merged.shape, h_node.shape)
            edge_emb = ch.functions.relu(
                self.edge_parameters[i](merged))
            h_edge = ch.functions.sparse_matmul(reduce_matrix, edge_emb,
                                                transa=True)
            h_node = ch.functions.relu(self.node_parameters[i](h_edge))

        return h_node

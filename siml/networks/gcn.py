
import torch

from . import abstract_gcn


class GCN(abstract_gcn.AbstractGCN):
    """Graph Convolutional network according to
    https://arxiv.org/abs/1609.02907 .
    """

    def __init__(self, block_setting):
        super().__init__(
            block_setting, create_subchain=True,
            residual=block_setting.residual)

    def _forward_single_core(self, x, subchain_index, support):
        h = x
        for subchain, dropout_ratio, activation in zip(
                self.subchains[subchain_index],
                self.dropout_ratios, self.activations):
            h = subchain(h)
            h = torch.nn.functional.dropout(
                h, p=dropout_ratio, training=self.training)
            h = activation(torch.sparse.mm(support, h))
        return h

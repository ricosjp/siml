
import torch

from . import header


class GCN(header.AbstractGCN):
    """Graph Convolutional network according to
    https://arxiv.org/abs/1609.02907 .
    """

    def __init__(self, block_setting, residual=False):
        super().__init__(
            block_setting, create_subchain=True, residual=residual)

    def _forward_single_core(self, x, subchain_index, support):
        h = x
        for subchain, activation in zip(
                self.subchains[subchain_index], self.activations):
            h = subchain(h)
            h = activation(torch.sparse.mm(support, h))
        return h


class ResGCN(GCN):
    """Residual version of Graph Convolutional network.
    """

    def __init__(self, block_setting):
        super().__init__(block_setting, residual=True)

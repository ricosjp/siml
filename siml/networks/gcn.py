
import torch
import torch.nn.functional as functional

from . import header


class GCN(header.AbstractGCN):
    """Graph Convolutional network according to
    https://arxiv.org/abs/1609.02907 .
    """

    def _call_single(self, x, support):
        h = x
        for link, dropout_ratio, activation in zip(
                self, self.dropout_ratios, self.activations):
            h = torch.einsum('mf,gf->mg', h, link.weight) + link.bias
            h = functional.dropout(h, p=dropout_ratio, training=self.training)
            h = activation(torch.sparse.mm(support, h))
        return h


class ResGCN(header.AbstractGCN):
    """Residual version of Graph Convolutional network.
    """

    def __init__(self, block_setting):
        super().__init__(block_setting)
        self.activations = self.activations[:-1] \
            + [header.DICT_ACTIVATIONS['identity']] + [self.activations[-1]]
        nodes = block_setting.nodes
        if nodes[0] == nodes[-1]:
            self.shortcut = header.identity
        else:
            self.shortcut = torch.nn.Linear(nodes[0], nodes[-1])

    def _call_single(self, x, support):
        h = x
        for link, dropout_ratio, activation in zip(
                self.subchains, self.dropout_ratios, self.activations):
            h = torch.einsum('mf,gf->mg', h, link.weight) + link.bias
            h = functional.dropout(h, p=dropout_ratio, training=self.training)
            h = activation(torch.sparse.mm(support, h))

        return self.activations[-1](h + self.shortcut(x))

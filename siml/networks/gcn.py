import chainer as ch

from . import header


class GCN(header.AbstractGCN):
    """Graph Convolutional network according to
    https://arxiv.org/abs/1609.02907 .
    """

    def _call_single(self, x, support):
        h = x
        for link, dropout_ratio, activation in zip(
                self, self.dropout_ratios, self.activations):
            h = ch.functions.einsum('mf,gf->mg', h, link.W) + link.b
            h = ch.functions.dropout(h, ratio=dropout_ratio)
            h = activation(ch.functions.sparse_matmul(support, h))
        return h


class ResGCN(header.AbstractGCN):
    """Residual version of Graph Convolutional network.
    """

    def __init__(self, block_setting):
        super().__init__(block_setting)
        nodes = block_setting.nodes
        with self.init_scope():
            self.linear = ch.links.Linear(nodes[0], nodes[-1])

    def _call_single(self, x, support):
        h = x
        for link, dropout_ratio, activation in zip(
                self.subchains, self.dropout_ratios, self.activations):
            h = ch.functions.einsum('mf,gf->mg', h, link.W) + link.b
            h = ch.functions.dropout(h, ratio=dropout_ratio)
            h = activation(ch.functions.sparse_matmul(support, h))

        return h + self.activations[-1](self.linear(x))

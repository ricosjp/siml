
import torch

from . import abstract_gcn


class GradGCN(abstract_gcn.AbstractGCN):
    """Graph Convolutional network with taking into account spatial gradient.
    """

    def __init__(self, block_setting):
        len_support = len(block_setting.support_input_indices)
        if len_support < 2:
            raise ValueError(
                'len(support_input_indices) should be larger than 1 '
                f"({len_support} found.)")
        super().__init__(
            block_setting, create_subchain=True, multiple_networks=False,
            residual=block_setting.residual)
        return

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

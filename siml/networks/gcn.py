
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

        self.factor = block_setting.optional.get(
            'factor', 1.)
        print(f"Factor: {self.factor}")
        self.ah_w = block_setting.optional.get(
            'ah_w', False)
        if self.ah_w:
            print(f"Matrix multiplication mode: (AH) W")
        else:
            print(f"Matrix multiplication mode: A (HW)")

        return

    def _forward_single_core(self, x, subchain_index, support):
        h = x
        for subchain, dropout_ratio, activation in zip(
                self.subchains[subchain_index],
                self.dropout_ratios, self.activations):

            if self.ah_w:
                # Pattern A: (A H) W
                support = support.to(h.device)
                h = subchain(self.factor * torch.sparse.mm(support, h))

            else:
                # Pattern B: A (H W)
                support = support.to(subchain.weight.device)
                h = torch.sparse.mm(support, subchain(self.factor * h))

            h = torch.nn.functional.dropout(
                h, p=dropout_ratio, training=self.training)
            h = activation(h)
        return h

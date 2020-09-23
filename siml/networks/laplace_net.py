
import torch

from . import abstract_gcn


class LaplaceNet(abstract_gcn.AbstractGCN):
    """Graph Convolutional network with laplacian.
    """

    def __init__(self, block_setting):
        len_support = len(block_setting.support_input_indices)
        if len_support < 2:
            raise ValueError(
                'len(support_input_indices) should be larger than 1 '
                f"({len_support} found.)")
        if block_setting.optional.get('gather_function', 'sum') != 'sum':
            raise ValueError(
                f"Cannot set gather_function for: {block_setting}")
        super().__init__(
            block_setting, create_subchain=True, multiple_networks=False,
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

    def _forward_single(self, x, supports):
        if self.residual:
            h_res = self._forward_single_core(x, supports) + self.shortcut(x)
            return self.activations[-1](h_res)
        else:
            return self._forward_single_core(x, supports)

    def _forward_single_core(self, x, supports):
        h = x
        for subchain, dropout_ratio, activation in zip(
                self.subchains[0],
                self.dropout_ratios, self.activations):

            if self.ah_w:
                # Pattern A: (GX (GX H) + GY (GY H) + GZ (GZ H)) W
                h = subchain(torch.sum(torch.stack([
                    torch.sparse.mm(support, torch.sparse.mm(support, h))
                    * self.factor**2
                    for support in supports]), dim=0))

            else:
                # Pattern B: GX (GX (H W)) + GY (GY (H W)) + GZ (GZ (H W))
                h = subchain(self.factor * h)
                h = torch.sum(torch.stack([
                    torch.sparse.mm(support, torch.sparse.mm(support, h))
                    for support in supports]) ** self.factor**2, dim=0)

            h = torch.nn.functional.dropout(
                h, p=dropout_ratio, training=self.training)
            h = activation(h)
        return h

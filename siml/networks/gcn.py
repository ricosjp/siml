
import torch

from . import abstract_gcn


class GCN(abstract_gcn.AbstractGCN):
    """Graph Convolutional network based on https://arxiv.org/abs/1609.02907 .
    """

    @staticmethod
    def get_name():
        return 'gcn'

    def __init__(self, block_setting):
        self.create_subchains_flag = block_setting.optional.get(
            'create_subchains', True)
        super().__init__(
            block_setting,
            create_subchain=self.create_subchains_flag,
            residual=block_setting.residual)

        self.factor = block_setting.optional.get(
            'factor', 1.)
        self.repeat = block_setting.optional.get(
            'repeat', 1)
        self.convergence_threshold = block_setting.optional.get(
            'convergence_threshold', None)
        print(f"Factor: {self.factor}")
        print(
            f"max repeat: {self.repeat}, "
            f"convergeence threshold: {self.convergence_threshold}")
        self.ah_w = block_setting.optional.get(
            'ah_w', False)
        if self.ah_w:
            print("Matrix multiplication mode: (AH) W")
        else:
            print("Matrix multiplication mode: A (HW)")

        return

    def _forward_single_core(self, x, subchain_index, support):
        h = x

        if not self.create_subchains_flag:
            h = self._propagate(x, support)
            return h

        for subchain, dropout_ratio, activation in zip(
                self.subchains[subchain_index],
                self.dropout_ratios, self.activations):

            if self.ah_w:
                # Pattern A: (A H) W
                h = subchain(self._propagate(h, support))

            else:
                # Pattern B: A (H W)
                h = self._propagate(subchain(h), support)

            h = torch.nn.functional.dropout(
                h, p=dropout_ratio, training=self.training)
            h = activation(h)
        return h

    def _propagate(self, x, support):
        result_shape = list(x.shape)
        h = torch.reshape(x, (result_shape[0], -1))
        result_shape[0] = support.shape[0]
        for _ in range(self.repeat):
            h_previous = h
            h = torch.sparse.mm(support, h) * self.factor
            if self.convergence_threshold is not None:
                residual = torch.linalg.norm(
                    h - h_previous) / (torch.linalg.norm(h_previous) + 1.e-5)
                if residual < self.convergence_threshold:
                    break
        return torch.reshape(h, result_shape)

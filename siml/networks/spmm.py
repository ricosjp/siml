
import torch

from . import abstract_gcn


class SpMM(abstract_gcn.AbstractGCN):
    """Layer to compute sparse matrix times dence matrix.
    """

    @staticmethod
    def get_name():
        return 'spmm'

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return True

    def __init__(self, block_setting):
        super().__init__(
            block_setting, create_subchain=False, residual=False)

        self.factor = block_setting.optional.get(
            'factor', 1.)
        self.mode = block_setting.optional.get(
            'mode', 'sum')
        self.transpose = self.block_setting.optional.get('transpose', False)
        print(f"Factor: {self.factor}")

        return

    def _forward_single_core(self, x, subchain_index, support):
        result_shape = list(x.shape)
        h = torch.reshape(x, (result_shape[0], -1))
        if self.transpose:
            support = support.transpose(0, 1)
        result_shape[0] = support.shape[0]

        h = torch.sparse.mm(support, h) * self.factor

        if self.mode == 'sum':
            scales = 1.
        elif self.mode == 'mean':
            scales = 1 / torch.sparse.sum(support, dim=1).to_dense()[..., None]
        else:
            raise ValueError(f"Unexpected mode: {self.mode}")
        h = scales * h

        return torch.reshape(h, result_shape)

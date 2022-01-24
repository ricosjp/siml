
import torch

from . import abstract_equvariant_gnn
from . import mlp
from . import tensor_operations


class PENN(abstract_equvariant_gnn.AbstractEquivariantGNN):
    """Physics-Embedded Neural Network."""

    @staticmethod
    def get_name():
        return 'penn'

    @staticmethod
    def accepts_multiple_inputs():
        return True

    def __init__(self, block_setting):
        if block_setting.nodes[0] == block_setting.nodes[-1]:
            block_setting.optional['create_subchain'] = False

        super().__init__(block_setting)

        if len(self.propagation_functions) != 1:
            raise ValueError(
                f"Set only one propagation function for: {block_setting}")

        if self.block_setting.optional['propagations'] == 'convolution':
            self.mlp = mlp.MLP(self.block_setting)
        else:
            self.mlp = tensor_operations.EquivariantMLP(self.block_setting)
        return

    def _convolution(self, x, supports, top=True):
        """Calculate convolution G \\ast x.

        Parameters
        ----------
        x: torch.Tensor
            [n_vertex, n_feature]-shaped tensor.
        supports: list[torch.Tensor]
            - 0: [n_edge, n_vertex]-shaped spatial graph gradient incidence
              matrix.
            - 1, 2, 3: [n_vertex, n_edge]-shaped edge integration incidence
              matrix.

        Returns
        -------
        y: torch.Tensor
            [n_vertex, dim, n_feature]-shaped tensor.
        """
        if self.support_tensor_rank != 1:
            raise NotImplementedError(
                f"Invalid support_tensor_rank: {self.support_tensor_rank}")

        grad_inc = supports[0]
        int_incs = supports[1:]

        edge = self.mlp(grad_inc.mm(x))
        h = torch.stack([int_inc.mm(edge) for int_inc in int_incs], axis=-2)
        return h

    def _tensor_product(self, x, supports, top=True):
        """Calculate tensor product G \\otimes x.

        Parameters
        ----------
        x: torch.Tensor
            [n_vertex, dim, dim, ..., n_feature]-shaped tensor.
                       ~~~~~~~~~~~~~~
                       tensor rank repetition
        supports: list[torch.Tensor]
            - 0: [n_edge, n_vertex]-shaped spatial graph gradient incidence
              matrix.
            - 1, 2, 3: [n_vertex, n_edge]-shaped edge integration incidence
              matrix.

        Returns
        -------
        y: torch.Tensor
            [n_vertex, dim, dim, ..., dim, n_feature]-shaped tensor.
                       ~~~~~~~~~~~~~~~~~~~
                       tensor rank+1 repetition
        """
        shape = x.shape
        dim = len(supports)
        tensor_rank = len(shape) - 2
        if tensor_rank == 0:
            h = self._convolution(x, supports, top=False)
        elif tensor_rank > 0:
            h = torch.stack([
                self._tensor_product(x[:, i_dim], supports, top=False)
                for i_dim in range(dim)], dim=-2)
        else:
            raise ValueError(f"Tensor shape invalid: {shape}")

        return h

    def _contraction(self, x, supports):
        """Calculate contraction G \\cdot B. It calculates
        \\sum_l G_{i,j,k_1,k_2,...,l} H_{jk_1,k_2,...,l,f}

        Parameters
        ----------
        x: torch.Tensor
            [n_vertex, dim, dim, ..., n_feature]-shaped tensor.
                       ~~~~~~~~~~~~~~
                       tensor rank repetition
        supports: list[torch.Tensor]
            - 0: [n_edge, n_vertex]-shaped spatial graph gradient incidence
              matrix.
            - 1, 2, 3: [n_vertex, n_edge]-shaped edge integration incidence
              matrix.

        Returns
        -------
        y: torch.Tensor
            [n_vertex, dim, ..., n_feature]-shaped tensor.
                       ~~~~~~~~~
                       tensor rank - 1 repetition
        """
        shape = x.shape
        tensor_rank = len(shape) - 2
        if tensor_rank == self.support_tensor_rank:
            if self.support_tensor_rank == 1:
                return torch.sum(torch.stack([
                    supports[i_dim].mm(x[:, i_dim]) for i_dim
                    in range(self.dim)]), dim=0)
            else:
                raise NotImplementedError(
                    f"Invalid support_tensor_rank: {self.support_tensor_rank}")
        elif tensor_rank > 1:
            return torch.stack([
                self._contraction(x[:, i_dim], supports)
                for i_dim in range(self.dim)], dim=1)
        else:
            raise ValueError(f"Tensor rank is 0 (shape: {shape})")

    def _rotation(self, x, supports):
        """Calculate rotation G \\times x.

        Parameters
        ----------
        x: torch.Tensor
            [n_vertex, dim, n_feature]-shaped tensor.
        supports: list[torch.Tensor]
            - 0: [n_edge, n_vertex]-shaped spatial graph gradient incidence
              matrix.
            - 1, 2, 3: [n_vertex, n_edge]-shaped edge integration incidence
              matrix.

        Returns
        -------
        y: torch.Tensor
            [n_vertex, dim, n_feature]-shaped tensor.
        """
        shape = x.shape
        dim = len(supports)
        tensor_rank = len(shape) - 2
        if tensor_rank != 1:
            raise ValueError(f"Tensor shape invalid: {shape}")
        if dim != 3:
            raise ValueError(f"Invalid dimension: {dim}")
        h = torch.stack([
            supports[1].mm(x[:, 2]) - supports[2].mm(x[:, 1]),
            supports[2].mm(x[:, 0]) - supports[0].mm(x[:, 2]),
            supports[0].mm(x[:, 1]) - supports[1].mm(x[:, 0]),
        ], dim=-2)

        return h

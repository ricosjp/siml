
import torch

from . import abstract_equivariant_gnn
from . import sparse


class IsoGCN(abstract_equivariant_gnn.AbstractEquivariantGNN):
    """IsoGCN according to https://arxiv.org/abs/2005.06316 ."""

    @staticmethod
    def get_name():
        return 'iso_gcn'

    @staticmethod
    def accepts_multiple_inputs():
        return True

    def _tensor_product(self, x, *args, supports):
        """Calculate tensor product G \\otimes x.

        Parameters
        ----------
        x: torch.Tensor
            [n_vertex, dim, dim, ..., n_feature]-shaped tensor.
                       ~~~~~~~~~~~~~~
                       tensor rank repetition
        supports: list[torch.Tensor]
            List of [n_vertex, n_vertex]-shaped sparse tensor.

        Returns
        -------
        y: torch.Tensor
            [n_vertex, dim, dim, ..., dim, n_feature]-shaped tensor.
                       ~~~~~~~~~~~~~~~~~~~
                       tensor rank+1 repetition
        """
        shape = x.shape
        tensor_rank = len(shape) - 2
        if tensor_rank < 0:
            raise ValueError(f"Tensor shape invalid: {shape}")

        h = torch.stack([
            sparse.mul(support, x) for support in supports], dim=1)

        return h

    def _convolution(self, x, *args, supports):
        """Calculate convolution G \\ast x.

        Parameters
        ----------
        x: torch.Tensor
            [n_vertex, n_feature]-shaped tensor.
        supports: list[torch.Tensor]
            List of [n_vertex, n_vertex]-shaped sparse tensor.

        Returns
        -------
        y: torch.Tensor
            [n_vertex, dim, n_feature]-shaped tensor.
        """
        return self._tensor_product(x, *args, supports=supports)

    def _contraction(self, x, *args, supports):
        """Calculate contraction G \\cdot B. It calculates
        \\sum_l G_{i,j,k_1,k_2,...,l} H_{jk_1,k_2,...,l,f}

        Parameters
        ----------
        x: torch.Tensor
            [n_vertex, dim, dim, ..., n_feature]-shaped tensor.
                       ~~~~~~~~~~~~~~
                       tensor rank repetition
        supports: list[torch.Tensor]
            List of [n_vertex, n_vertex]-shaped sparse tensor.

        Returns
        -------
        y: torch.Tensor
            [n_vertex, dim, ..., n_feature]-shaped tensor.
                       ~~~~~~~~~
                       tensor rank - 1 repetition
        """
        higher_h = self._tensor_product(x, *args, supports=supports)
        return torch.einsum('ikk...->i...', higher_h)

    def _rotation(self, x, *args, supports):
        """Calculate rotation G \\times x.

        Parameters
        ----------
        x: torch.Tensor
            [n_vertex, dim, n_feature]-shaped tensor.
        supports: list[torch.Tensor]
            List of [n_vertex, n_vertex]-shaped sparse tensor.

        Returns
        -------
        y: torch.Tensor
            [n_vertex, dim, n_feature]-shaped tensor.
        """
        h = self._tensor_product(x, *args, supports=supports)
        return torch.stack([
            h[:, 1, 2] - h[:, 2, 1],
            h[:, 2, 0] - h[:, 0, 2],
            h[:, 0, 1] - h[:, 1, 0],
        ], dim=1)

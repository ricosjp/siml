
import einops
import torch

from . import abstract_equivariant_gnn


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
        dim = len(supports)
        tensor_rank = len(shape) - 2
        if tensor_rank == 0:
            h = self._convolution(x, supports=supports)
        elif tensor_rank > 0:
            h = torch.stack([
                self._tensor_product(x[:, i_dim], supports=supports)
                for i_dim in range(dim)], dim=-2)
        else:
            raise ValueError(f"Tensor shape invalid: {shape}")

        return h

    def _convolution(self, x, *args, supports, top=True):
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
        if self.support_tensor_rank == 1:
            h = torch.stack([support.mm(x) for support in supports], axis=-2)
        elif self.support_tensor_rank == 2:
            n = x.shape[0]
            f = x.shape[-1]
            h = torch.reshape(
                torch.stack([support.mm(x) for support in supports], axis=-2),
                (n, self.dim, self.dim, f))
            if self.symmetric:
                h = (h + einops.rearrange(
                    h, 'element x1 x2 feature -> element x2 x1 feature')) / 2
        else:
            raise NotImplementedError

        return h

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
        shape = x.shape
        tensor_rank = len(shape) - 2
        if tensor_rank == self.support_tensor_rank:
            if self.support_tensor_rank == 1:
                return torch.sum(torch.stack([
                    supports[i_dim].mm(x[:, i_dim]) for i_dim
                    in range(self.dim)]), dim=0)
            elif self.support_tensor_rank == 2:
                # ijkm, jnmf -> iknf
                return torch.stack([
                    torch.stack([
                        torch.sum(torch.stack([
                            supports[self.dim*k+m].mm(x[:, n, m])
                            for m in range(self.dim)]), dim=0)
                        for n in range(self.dim)], dim=1)
                    for k in range(self.dim)], dim=1)
            else:
                raise ValueError
        elif tensor_rank > 1:
            return torch.stack([
                self._contraction(x[:, i_dim], supports=supports)
                for i_dim in range(self.dim)], dim=-2)
        else:
            raise ValueError(f"Tensor rank is 0 (shape: {shape})")

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

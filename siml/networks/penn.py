
import numpy as np
import torch

from . import abstract_equivariant_gnn
from . import mlp
from . import sparse
from . import tensor_operations


class PENN(abstract_equivariant_gnn.AbstractEquivariantGNN):
    """Physics-Embedded Neural Network."""

    @staticmethod
    def get_name():
        return 'penn'

    @staticmethod
    def accepts_multiple_inputs():
        return True

    @classmethod
    def _get_n_input_node(
            cls, block_setting, predecessors, dict_block_setting,
            input_length, **kwargs):
        return np.max([
            dict_block_setting[predecessor].nodes[-1]
            for predecessor in predecessors])

    def __init__(self, block_setting):
        block_setting.optional['create_subchain'] = False
        block_setting.optional['set_last_activation'] = False

        super().__init__(block_setting)

        if len(self.propagation_functions) != 1:
            raise ValueError(
                f"Set only one propagation function for: {block_setting}")

        self.use_mlp = self.block_setting.optional.get('use_mlp', None)

        if self.use_mlp is None:
            if self.block_setting.optional['propagations'] == 'contraction':
                self.mlp = mlp.MLP(self.block_setting)
            else:
                self.mlp = tensor_operations.EquivariantMLP(
                    self.block_setting)
        elif self.use_mlp:
            self.mlp = mlp.MLP(self.block_setting)
        else:
            self.mlp = tensor_operations.EquivariantMLP(
                self.block_setting)

        return

    def _convolution(self, x, *args, supports, top=True):
        """Calculate convolution G \\ast x.

        Parameters
        ----------
        x: torch.Tensor
            [n_vertex, n_feature]-shaped tensor.
        inversed_moment_tensors: torch.Tensor
            [n_vertex, 3, 3, 1]-shaped inversed moment tensors.
        supports: list[torch.Tensor]
            - 0, 1, 2: [n_edge, n_vertex]-shaped spatial graph gradient
              incidence matrix.
            - 3: [n_vertex, n_edge]-shaped edge integration incidence
              matrix.

        Returns
        -------
        y: torch.Tensor
            [n_vertex, dim, n_feature]-shaped tensor.
        """
        return self._tensor_product(
            x, *args, supports=supports)

    def _tensor_product(self, x, *args, supports, top=True):
        """Calculate tensor product G \\otimes x.

        Parameters
        ----------
        x: torch.Tensor
            [n_vertex, dim, dim, ..., n_feature]-shaped tensor.
                       ~~~~~~~~~~~~~~
                       tensor rank repetition
        inversed_moment_tensors: torch.Tensor
            [n_vertex, 3, 3, 1]-shaped inversed moment tensors.
        supports: list[torch.Tensor]
            - 0, 1, 2: [n_edge, n_vertex]-shaped spatial graph gradient
              incidence matrix.
            - 3: [n_vertex, n_edge]-shaped edge integration incidence
              matrix.

        Returns
        -------
        y: torch.Tensor
            [n_vertex, dim, dim, ..., dim, n_feature]-shaped tensor.
                       ~~~~~~~~~~~~~~~~~~~
                       tensor rank+1 repetition
        """
        if self.support_tensor_rank != 1:
            raise NotImplementedError(
                f"Invalid support_tensor_rank: {self.support_tensor_rank}")
        if len(supports) != 4:
            raise ValueError(
                'Invalid length of supports '
                f"({len(supports)} given, expected 4)")
        inversed_moment_tensors = args[0]

        grad_incs = supports[:3]
        int_inc = supports[3]

        edge = self.mlp(torch.stack([
            sparse.mul(grad_inc, x) for grad_inc in grad_incs], axis=1))
        h = torch.einsum(
            'ikl,il...f->ik...f', inversed_moment_tensors[..., 0],
            sparse.mul(int_inc, edge))
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
        inversed_moment_tensors: torch.Tensor
            [n_vertex, 3, 3, 1]-shaped inversed moment tensors.
        supports: list[torch.Tensor]
            - 0, 1, 2: [n_edge, n_vertex]-shaped spatial graph gradient
              incidence matrix.
            - 3: [n_vertex, n_edge]-shaped edge integration incidence
              matrix.

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
        inversed_moment_tensors: torch.Tensor
            [n_vertex, 3, 3, 1]-shaped inversed moment tensors.
        supports: list[torch.Tensor]
            - 0, 1, 2: [n_edge, n_vertex]-shaped spatial graph gradient
              incidence matrix.
            - 3: [n_vertex, n_edge]-shaped edge integration incidence
              matrix.

        Returns
        -------
        y: torch.Tensor
            [n_vertex, dim, n_feature]-shaped tensor.
        """
        shape = x.shape
        dim = len(supports) - 1
        tensor_rank = len(shape) - 2
        if tensor_rank != 1:
            raise ValueError(f"Tensor shape invalid: {shape}")
        if dim != 3:
            raise ValueError(f"Invalid dimension: {dim}")
        inversed_moment_tensors = args[0]

        t = self._tensor_product(
            x, inversed_moment_tensors, supports=supports)
        h = torch.stack([
            t[:, 1, 2] - t[:, 2, 1],
            t[:, 2, 0] - t[:, 0, 2],
            t[:, 0, 1] - t[:, 1, 0],
        ], dim=-2)

        return h

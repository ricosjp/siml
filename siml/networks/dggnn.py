
import copy

import numpy as np
import torch

from . import abstract_equivariant_gnn
from . import identity
from . import mlp
from . import sparse
from . import tensor_operations


class DGGNN(abstract_equivariant_gnn.AbstractEquivariantGNN):
    """DGGNN block."""

    @staticmethod
    def get_name():
        return 'dggnn'

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
        self.trainable = self.block_setting.optional.get('trainable', True)
        self.riemann_solver = self.block_setting.optional.get(
            'riemann_solver', None)

        self.mlp = self._create_mlp(trainable=self.trainable)
        if self.riemann_solver is not None:
            if self.riemann_solver == 'lax_friedrichs':
                bias = False

            self.solver_mlp = self._create_mlp(
                trainable=self.trainable, bias=bias)

        return

    def _create_mlp(self, trainable, bias=None):
        block_setting = copy.copy(self.block_setting)

        if not trainable:
            return identity.Identity(block_setting)

        if bias is not None:
            block_setting.bias = bias
            print(f"Bias set to {bias=}")

        if self.use_mlp is None:
            if block_setting.optional['propagations'] == 'contraction':
                return mlp.MLP(block_setting)
            else:
                return tensor_operations.EquivariantMLP(block_setting)

        if self.use_mlp:
            return mlp.MLP(block_setting)
        else:
            return tensor_operations.EquivariantMLP(block_setting)

    def _separate_supports(self, supports):
        """Separate supports matrices to the desired format.

        Parameters
        ----------
        supports: list[torch.Tensor]
            - 0: [n_facet, n_cell]-shaped signed incidence matrix.
            - 1, 2, ...: [n_cell, n_facet]-shaped area-weighted normal
              incidence matrix.

        Returns
        -------
        torch.sparse_coo_tensor:
            [n_facet, n_cell]-shaped signed incidence matrix.
        list[torch.sparse_coo_tensor]:
            [n_cell, n_facet]-shaped area-weighted normal
            incidence matrices.
        """
        if self.support_tensor_rank != 1:
            raise NotImplementedError(
                f"Invalid support_tensor_rank: {self.support_tensor_rank}")
        if len(supports) < 3:
            raise ValueError(
                'Invalid length of supports '
                f"({len(supports)} given, expected >= 3)")

        signed_inc_cell2facet = supports[0]
        area_normal_inc_facet2cell = supports[1:]

        return signed_inc_cell2facet, area_normal_inc_facet2cell

    def _spread(self, cell_x, *args, supports, top=True):
        """Spread cell features to facet features.

        Parameters
        ----------
        cell_x: torch.Tensor
            [n_cell, dim, dim, ..., n_feature]-shaped tensor.
                     ~~~~~~~~~~~~~~
                     tensor rank repetition
        supports: list[torch.Tensor]
            - 0: [n_facet, n_cell]-shaped signed incidence matrix.
            - 1, 2, ...: [n_cell, n_facet]-shaped area-weighted normal
              incidence matrix.

        Returns
        -------
        facet_y: torch.Tensor
            [n_facet, dim, dim, ..., n_feature]-shaped tensor.
                      ~~~~~~~~~~~~~~
                      tensor rank repetition
        """
        signed_inc_cell2facet, area_normal_inc_facet2cell \
            = self._separate_supports(supports)

        inc_facet2cell = torch.abs(signed_inc_cell2facet)
        scale = 1 / torch.sparse.sum(inc_facet2cell, dim=1).to_dense()
        ave_facet_values = self.mlp(
            torch.einsum(
                'i,i...->i...', scale, sparse.mul(inc_facet2cell, cell_x)))

        if self.riemann_solver == 'lax_friedrichs':
            # Lax--Friedrichs method
            diff_facet_values = self.solver_mlp(
                signed_inc_cell2facet.mm(cell_x) / 2)
            # TODO: Determine CFL number
            facet_values = ave_facet_values + diff_facet_values
        else:
            facet_values = ave_facet_values
        return facet_values

    def _convolution(self, facet_x, *args, supports, top=True):
        """Calculate convolution to compute gradient.

        Parameters
        ----------
        x: torch.Tensor
            [n_facet, n_feature]-shaped tensor.
        supports: list[torch.Tensor]
            - 0: [n_facet, n_cell]-shaped signed incidence matrix.
            - 1, 2, ...: [n_cell, n_facet]-shaped area-weighted normal
              incidence matrix.

        Returns
        -------
        y: torch.Tensor
            [n_cell, dim, n_feature]-shaped tensor.
        """
        return self._tensor_product(
            facet_x, *args, supports=supports)

    def _tensor_product(self, facet_x, *args, supports, top=True):
        """Calculate tensor product to compute (higher-rank) gradient.

        Parameters
        ----------
        facet_x: torch.Tensor
            [n_facet, dim, dim, ..., n_feature]-shaped tensor.
                      ~~~~~~~~~~~~~~
                      tensor rank repetition
        supports: list[torch.Tensor]
            - 0: [n_facet, n_cell]-shaped signed incidence matrix.
            - 1, 2, ...: [n_cell, n_facet]-shaped area-weighted normal
              incidence matrix.

        Returns
        -------
        cell_y: torch.Tensor
            [n_cell, dim, dim, ..., dim, n_feature]-shaped tensor.
                     ~~~~~~~~~~~~~~~~~~~
                     tensor rank+1 repetition
        """
        signed_inc_cell2facet, area_normal_inc_facet2cell \
            = self._separate_supports(supports)

        cell_h = self.mlp(torch.stack([
            sparse.mul(grad_inc, facet_x)
            for grad_inc in area_normal_inc_facet2cell], axis=1))
        return cell_h

    def _contraction(self, facet_x, *args, supports):
        """Calculate contraction to compute divergence.

        Parameters
        ----------
        facet_x: torch.Tensor
            [n_facet, dim, dim, ..., n_feature]-shaped tensor.
                      ~~~~~~~~~~~~~~
                      tensor rank repetition
        supports: list[torch.Tensor]
            - 0: [n_facet, n_cell]-shaped signed incidence matrix.
            - 1, 2, ...: [n_cell, n_facet]-shaped area-weighted normal
              incidence matrix.

        Returns
        -------
        cell_y: torch.Tensor
            [n_cell, dim, ..., n_feature]-shaped tensor.
                     ~~~~~~~~~
                     tensor rank - 1 repetition
        """
        cell_h = self._tensor_product(facet_x, *args, supports=supports)
        # raise ValueError(cell_h[:10, :, :, 0])
        cell_h = torch.einsum('ikk...->i...', cell_h)
        # print(facet_x.shape)
        # raise ValueError(
        #     facet_x[:10, 0, 0], cell_h[:10, 0],
        #     supports[1].coalesce().values()[:10])
        # print(
        #     facet_x[:10, 0, 0], cell_h[:10, 0],
        #     supports[1].coalesce().values()[:10])
        return cell_h

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

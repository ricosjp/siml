
import einops
import torch

from . import abstract_gcn
from . import identity


class IsoGCN(abstract_gcn.AbstractGCN):
    """IsoGCN according to https://arxiv.org/abs/2005.06316 ."""

    def __init__(self, block_setting):
        self.is_first = True  # For block setting validation

        len_support = len(block_setting.support_input_indices)
        if len_support < 2:
            raise ValueError(
                'len(support_input_indices) should be larger than 1 '
                f"({len_support} found.)")

        self.create_subchain = block_setting.optional.get(
            'create_subchain', True)
        super().__init__(
            block_setting, create_subchain=self.create_subchain,
            multiple_networks=False, residual=block_setting.residual)
        if not self.create_subchain:
            print(f"Skipe subchain creation for: {block_setting.name}")
            self.subchains = [[identity.Identity(block_setting)]]

        if len(self.subchains[0]) > 1:
            raise ValueError(
                f"# of layers should be 0 or 1 for: {block_setting}")

        self.dim = block_setting.optional.get('dim', 3)
        self.support_tensor_rank = block_setting.optional.get(
            'support_tensor_rank', 1)
        self.symmetric = block_setting.optional.get('symmetric', False)
        if self.symmetric:
            print(f"Output symmetric matrix for: {block_setting.name}")

        self.merge_sparse = block_setting.optional.get('merge_sparse', False)
        print(f"Merge sparse: {self.merge_sparse}")
        self.factor = block_setting.optional.get('factor', 1.)
        print(f"Factor: {self.factor}")
        self.ah_w = block_setting.optional.get(
            'ah_w', False)
        if self.ah_w:
            print(f"Matrix multiplication mode: (AH) W")
        else:
            print(f"Matrix multiplication mode: A (HW)")

        if 'propagations' not in block_setting.optional:
            raise ValueError(
                f"Specify 'propagations' in optional for: {block_setting}")

        self.propagation_functions = self._create_propagation_functions()

        return

    def _validate_block(self, x, supports):
        shape = x.shape
        self.x_tensor_rank \
            = len(shape) - 2  # (n_vertex, dim, dim, ..., n_feature)

        if len(supports) != self.dim**self.support_tensor_rank:
            raise ValueError(
                f"{self.dim**self.support_tensor_rank} ength of supports "
                f"expected (actual: {len(supports)} for: {self.block_setting}")

        if not self.create_subchain:
            return

        str_propagations = self.block_setting.optional['propagations']
        n_propagation = len(str_propagations)
        n_contraction = sum([
            s == 'contraction' for s in str_propagations])
        n_rank_raise_propagation = n_propagation - n_contraction
        estimated_output_tensor_rank = \
            self.x_tensor_rank - n_contraction + n_rank_raise_propagation

        if estimated_output_tensor_rank > 0 \
                and self.block_setting.activations[0] != 'identity':
            raise ValueError(
                'Set identity activation for rank '
                f"{estimated_output_tensor_rank} output: {self.block_setting}")

        if self.x_tensor_rank == 0:
            if estimated_output_tensor_rank > 0:
                # rank 0 -> rank k tensor
                if self.block_setting.bias and self.ah_w:
                    raise ValueError(
                        'Set bias = False for rank 0 -> k with (AH) W: '
                        f"{self.block_setting}")

        else:
            if estimated_output_tensor_rank == 0:
                # rank k -> rank 0 tensor
                if self.block_setting.bias and not self.ah_w:
                    raise ValueError(
                        'Set bias = False for rank k -> 0 with A (HW): '
                        f"{self.block_setting}")
            else:
                # rank k -> rank l tensor
                if self.block_setting.bias:
                    raise ValueError(
                        'Set bias = False for rank k -> l with A (HW): '
                        f"{self.block_setting}")
        return

    def _forward_single(self, x, merged_support):
        if self.is_first:
            self._validate_block(x, merged_support)
            self.is_first = False
        if self.residual:
            shortcut = self.shortcut(x)
        else:
            shortcut = 0.

        h_res = self._propagate(x, merged_support)
        if self.block_setting.activation_after_residual:
            h_res = self.activations[-1](h_res + shortcut)
        else:
            h_res = self.activations[-1](h_res) + shortcut

        return h_res

    def _propagate(self, x, support):
        h = x
        if not self.ah_w:
            # A (H W)
            h = self.subchains[0][0](h)

        for propagation_function in self.propagation_functions:
            h = propagation_function(h, support) * self.factor

        if self.ah_w:
            # (A H) W
            h = self.subchains[0][0](h)

        h = torch.nn.functional.dropout(
            h, p=self.dropout_ratios[0], training=self.training)
        return h

    def _create_propagation_functions(self):
        str_propagations = self.block_setting.optional['propagations']
        propagation_functions = [
            self._create_propagation_function(str_propagation)
            for str_propagation in str_propagations]

        return propagation_functions

    def _create_propagation_function(self, str_propagation):
        if self.merge_sparse:
            if str_propagation == 'convolution':
                return self._convolution_with_merge
            elif str_propagation == 'contraction':
                return self._contraction_with_merge
            elif str_propagation == 'tensor_product':
                return self._tensor_product_with_merge
            else:
                raise ValueError(
                    f"Unexpected propagation method: {str_propagation}")
        else:
            if str_propagation == 'convolution':
                return self._convolution_without_merge
            elif str_propagation == 'contraction':
                return self._contraction_without_merge
            elif str_propagation == 'tensor_product':
                return self._tensor_product_without_merge
            else:
                raise ValueError(
                    f"Unexpected propagation method: {str_propagation}")

    def _calculate_dim_without_merge(self, supports, n_vertex):
        dim, mod = divmod(supports.shape[0], n_vertex)
        if mod != 0:
            raise ValueError(
                'IsoGCN not supported for\n'
                f"    Sparse shape: {[s.shape for s in supports]}\n",
                f"    n_vertex: {n_vertex}")
        return dim

    def _tensor_product_without_merge(self, x, supports):
        """Calculate tensor product G \\otimes x.

        Parameters
        ----------
        x: torch.Tensor
            [n_vertex, dim, dim, ..., n_feature]-shaped tensor.
                       ~~~~~~~~~~~~~~
                       tensor rank repetition
        supports: List[torch.Tensor]
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
            h = self._convolution_without_merge(x, supports)
        elif tensor_rank > 0:
            h = torch.stack([
                self._convolution_without_merge(x[:, i_dim], supports)
                for i_dim in range(dim)], dim=-2)
        else:
            raise ValueError(f"Tensor shape invalid: {shape}")

        if self.symmetric:
            h = (h + einops.rearrange(
                h, 'element x1 x2 feature -> element x2 x1 feature')) / 2

        return h

    def _convolution_without_merge(self, x, supports, top=True):
        """Calculate convolution G \\ast x.

        Parameters
        ----------
        x: torch.Tensor
            [n_vertex, n_feature]-shaped tensor.
        supports: List[torch.Tensor]
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

    def _contraction_without_merge(self, x, supports):
        """Calculate contraction G \\cdot B. It calculates
        \\sum_l G_{i,j,k_1,k_2,...,l} H_{jk_1,k_2,...,l,f}

        Parameters
        ----------
        x: torch.Tensor
            [n_vertex, dim, dim, ..., n_feature]-shaped tensor.
                       ~~~~~~~~~~~~~~
                       tensor rank repetition
        supports: List[torch.Tensor]
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
                self._contraction_without_merge(x[:, i_dim], supports)
                for i_dim in range(self.dim)], dim=1)
        else:
            raise ValueError(f"Tensor rank is 0 (shape: {shape})")

    def _calculate_dim_with_merge(self, merged_support, n_vertex):
        dim, mod = divmod(merged_support.shape[0], n_vertex)
        if mod != 0:
            raise ValueError(
                'IsoGCN not supported for\n'
                f"    Sparse shape: {merged_support.shape}\n",
                f"    n_vertex: {n_vertex}")
        return dim

    def _convolution_with_merge(self, x, merged_support):
        """Calculate convolution G \\ast x.

        Parameters
        ----------
        x: torch.Tensor
            [n_vertex, n_feature]-shaped tensor.
        merged_support: torch.Tensor
            [n_vertex * dim, n_vertex * dim]-shaped sparse tensor.

        Returns
        -------
        y: torch.Tensor
            [n_vertex, dim, n_feature]-shaped tensor.
        """
        raise NotImplementedError
        n_vertex = len(x)
        dim = self._calculate_dim_with_merge(merged_support, n_vertex)
        x = torch.cat([x] * dim, dim=0)
        return merged_support.mm(x).view(n_vertex, dim, -1)

    def _contraction_with_merge(self, x, merged_support):
        """Calculate contraction G \\cdot x.

        Parameters
        ----------
        x: torch.Tensor
            [n_vertex, dim, n_feature]-shaped tensor.
        merged_support: torch.Tensor
            [n_vertex * dim, n_vertex * dim]-shaped sparse tensor.

        Returns
        -------
        y: torch.Tensor
            [n_vertex, n_feature]-shaped tensor.
        """
        raise NotImplementedError
        n_vertex = x.shape[1]
        dim = self._calculate_dim_with_merge(merged_support, n_vertex)
        return torch.sum(merged_support.mm(x.view(dim * n_vertex, -1)).view(
            dim, n_vertex, -1), dim=0)

    def _tensor_product_with_merge(self, x, merged_support):
        raise NotImplementedError

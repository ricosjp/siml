
import torch

from . import abstract_gcn


class IsoGCN(abstract_gcn.AbstractGCN):
    """IsoGCN according to https://arxiv.org/abs/2005.06316 ."""

    def __init__(self, block_setting):
        len_support = len(block_setting.support_input_indices)
        if len_support < 2:
            raise ValueError(
                'len(support_input_indices) should be larger than 1 '
                f"({len_support} found.)")
        super().__init__(
            block_setting, create_subchain=True, multiple_networks=False,
            residual=block_setting.residual)
        if len(self.subchains[0]) > 1:
            raise ValueError(
                f"# of layers should be 0 or 1 for: {block_setting}")

        self.factor = block_setting.optional.get(
            'factor', 1.)
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

    def _create_propagation_functions(self):
        str_propagations = self.block_setting.optional['propagations']
        propagation_functions = [
            self._create_propagation_function(str_propagation)
            for str_propagation in str_propagations]
        return propagation_functions

    def _create_propagation_function(self, str_propagation):
        if str_propagation == 'convolution':
            return self._convolution
        elif str_propagation == 'contraction':
            return self._contraction
        elif str_propagation == 'tensor_product':
            return self._tensor_product
        else:
            raise ValueError(
                f"Unexpected propagation method: {str_propagation}")

    def _calculate_dim(self, merged_support, n_vertex):
        dim, mod = divmod(merged_support.shape[0], n_vertex)
        if mod != 0:
            raise ValueError(
                'IsoGCN not supported for\n'
                f"    Sparse shape: {merged_support.shape}\n",
                f"    n_vertex: {n_vertex}")
        return dim

    def _convolution(self, x, merged_support):
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
            [dim, n_vertex, n_feature]-shaped tensor.
        """
        n_vertex = len(x)
        dim = self._calculate_dim(merged_support, n_vertex)
        x = torch.cat([x] * dim, dim=0)
        return merged_support.mm(x).view(dim, n_vertex, -1)

    def _contraction(self, x, merged_support):
        """Calculate contraction G \\cdot x.

        Parameters
        ----------
        x: torch.Tensor
            [dim, n_vertex, n_feature]-shaped tensor.
        merged_support: torch.Tensor
            [n_vertex * dim, n_vertex * dim]-shaped sparse tensor.

        Returns
        -------
        y: torch.Tensor
            [n_vertex, n_feature]-shaped tensor.
        """
        n_vertex = x.shape[1]
        dim = self._calculate_dim(merged_support, n_vertex)
        return torch.sum(merged_support.mm(x.view(dim * n_vertex, -1)).view(
            dim, n_vertex, -1), dim=0)

    def _tensor_product(self, x, merged_support):
        raise NotImplementedError

    def _forward_single(self, x, merged_support):
        if self.residual:
            shortcut = self.shortcut(x)
        else:
            shortcut = 0.

        h_res = self._propagate(x, merged_support)
        if self.block_setting.activation_after_residual:
            return self.activations[-1](h_res + shortcut)
        else:
            return self.activations[-1](h_res) + shortcut
        return

    def _propagate(self, x, merged_support):
        h = x
        if not self.ah_w:
            # A (H W)
            h = self.subchains[0][0](h)

        for propagation_function in self.propagation_functions:
            h = propagation_function(h, merged_support)

        if self.ah_w:
            # A (H W)
            h = self.subchains[0][0](self.factor * h)

        h = torch.nn.functional.dropout(
            h, p=self.dropout_ratios[0], training=self.training)
        return h

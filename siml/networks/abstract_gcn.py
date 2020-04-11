
import torch

from . import siml_module


class AbstractGCN(siml_module.SimlModule):

    def __init__(
            self, block_setting,
            *, create_subchain=True, residual=False, multiple_networks=None):
        """Initialize the NN.

        Parameters
        -----------
        block_setting: siml.setting.BlockSetting
            BlockSetting object.
        create_subchain: bool, optional [True]
            If True, create subchain to be trained.
        residual: bool, optional [False]
            If True, make the network residual.
        multiple_networks: bool, optional [None]
            If False, create only one subchain and share weights. If not set,
            use block_setting.optional['multiple_networks'] setting. If both
            not set, set True.
        """
        if multiple_networks is None:
            self.multiple_networks = block_setting.optional.get(
                'multiple_networks', True)
        else:
            self.multiple_networks = multiple_networks
        self.gather_function = block_setting.optional.get(
            'gather_function', 'sum')

        if self.gather_function == 'cat':
            len_support = len(block_setting.support_input_indices)
            if block_setting.nodes[-1] % len_support != 0:
                raise ValueError(
                    f"Set last node size multiple of {len_support} for: "
                    f"{block_setting}")
            overwritten_nodes = block_setting.nodes[:-1] + [
                block_setting.nodes[-1] // len_support]
        else:
            overwritten_nodes = block_setting.nodes

        super().__init__(
            block_setting, create_linears=False,
            residual_dimension=overwritten_nodes[-1])

        if create_subchain:
            self.subchains, self.subchain_indices = self._create_subchains(
                block_setting, nodes=overwritten_nodes)
        return

    def _create_subchains(
            self, block_setting, nodes, *,
            twice_input_nodes=False, square_weight=False, start_index=0):
        if self.multiple_networks:
            subchains = torch.nn.ModuleList([
                self._create_subsubchain(
                    nodes,
                    twice_input_nodes=twice_input_nodes,
                    square_weight=square_weight, start_index=start_index)
                for _ in block_setting.support_input_indices])
            subchain_indices = range(
                len(block_setting.support_input_indices))
        else:
            subchains = torch.nn.ModuleList([
                self._create_subsubchain(
                    nodes,
                    twice_input_nodes=twice_input_nodes,
                    square_weight=square_weight, start_index=start_index)])
            subchain_indices = [0] * len(
                block_setting.support_input_indices)

        return subchains, subchain_indices

    def _create_subsubchain(
            self, nodes, *,
            twice_input_nodes=False, square_weight=False, start_index=0):
        nodes = nodes[start_index:]
        if twice_input_nodes:
            factor = 2
        else:
            factor = 1

        if square_weight:
            node_tuples = [(n, n) for n in nodes]
        else:
            node_tuples = [
                (n1 * factor, n2) for n1, n2 in zip(nodes[:-1], nodes[1:])]

        return torch.nn.ModuleList([
            torch.nn.Linear(*node_tuple) for node_tuple in node_tuples])

    def forward(self, x, supports):
        """Execute the NN's forward computation.

        Parameters
        -----------
            x: numpy.ndarray or cupy.ndarray
                Input of the NN.
            supports: List[chainer.util.CooMatrix]
                List of support inputs.
        Returns
        --------
            y: numpy.ndarray of cupy.ndarray
                Output of the NN.
        """
        if len(x.shape) == 3:
            hs = torch.stack([
                self._forward_single(x_, supports_)
                for x_, supports_ in zip(x, supports)])
        else:
            hs = torch.stack([
                torch.stack([
                    self._forward_single(x__, supports_)
                    for x__, supports_ in zip(x_, supports)])
                for x_ in x])
        return hs

    def _forward_single(self, x, supports):
        if self.residual:
            if self.gather_function == 'sum':
                h_res = torch.sum(torch.stack([
                    self._forward_single_core(
                        x, self.subchain_indices[i], support)
                    for i, support in enumerate(supports)]), dim=0) \
                    + self.shortcut(x)

            elif self.gather_function == 'cat':
                h_res = torch.cat([
                    self._forward_single_core(
                        x, self.subchain_indices[i], support)
                    + self.shortcut(x)
                    for i, support in enumerate(supports)], dim=-1)

            else:
                raise ValueError(
                    f"Unknown gather_function: {self.gather_function}")

            return self.activations[-1](h_res)

        else:
            if self.gather_function == 'sum':
                return torch.sum(torch.stack([
                    self._forward_single_core(
                        x, self.subchain_indices[i], support)
                    for i, support in enumerate(supports)]), dim=0)

            elif self.gather_function == 'cat':
                return torch.cat([
                    self._forward_single_core(
                        x, self.subchain_indices[i], support)
                    for i, support in enumerate(supports)], dim=-1)

            else:
                raise ValueError(
                    f"Unknown gather_function: {self.gather_function}")

    def _forward_single_core(self, x, subchain_index, support):
        raise NotImplementedError

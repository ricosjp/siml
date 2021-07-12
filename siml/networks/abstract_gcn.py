
import torch

from . import siml_module


class AbstractGCN(siml_module.SimlModule):

    @staticmethod
    def is_trainable():
        return True

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return True

    def __init__(
            self, block_setting,
            *, create_subchain=True, residual=False, multiple_networks=None):
        """Initialize the NN.

        Parameters
        -----------
        block_setting: siml.setting.BlockSetting
            BlockSetting object.
        create_subchain: bool, optional
            If True, create subchain to be trained.
        residual: bool, optional
            If True, make the network residual.
        multiple_networks: bool, optional
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
        self.keep_self_loop = block_setting.optional.get(
            'keep_self_loop', False)

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
        else:
            self.subchain_indices = list(range(len(
                block_setting.support_input_indices)))

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
        bias = self.block_setting.bias
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
            torch.nn.Linear(*node_tuple, bias=bias)
            for node_tuple in node_tuples])

    def forward(self, x, supports, original_shapes=None):
        """Execute the NN's forward computation.

        Parameters
        -----------
        x: numpy.ndarray or cupy.ndarray
            Input of the NN.
        supports: list[chainer.util.CooMatrix]
            List of support inputs.

        Returns
        --------
            y: numpy.ndarray of cupy.ndarray
                Output of the NN.
        """
        if self.block_setting.time_series:
            hs = torch.stack([
                self._forward_single(_x, supports) for _x in x])
        else:
            hs = self._forward_single(x, supports)
        return hs

    def _forward_single(self, x, supports):
        if self.residual:
            if self.gather_function == 'sum':
                h_res = torch.sum(torch.stack([
                    self._forward_single_core(
                        x, self.subchain_indices[i], support)
                    for i, support in enumerate(supports)]), dim=0)
                shortcut = self.shortcut(x)

            elif self.gather_function == 'cat':
                h_res = torch.cat([
                    self._forward_single_core(
                        x, self.subchain_indices[i], support)
                    for i, support in enumerate(supports)], dim=-1)
                shortcut = torch.cat([
                    self.shortcut(x) for _ in supports], dim=-1)

            else:
                raise ValueError(
                    f"Unknown gather_function: {self.gather_function}")

            if self.block_setting.activation_after_residual:
                return self.activations[-1](h_res + shortcut)
            else:
                return self.activations[-1](h_res) + shortcut

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

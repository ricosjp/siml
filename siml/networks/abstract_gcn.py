
import torch

from . import siml_module


class AbstractGCN(siml_module.SimlModule):

    def __init__(
            self, block_setting,
            *, create_subchain=True, residual=False):
        """Initialize the NN.

        Parameters
        -----------
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
            create_subchain: bool, optional [True]
                If True, create subchain to be trained.
            residual: bool, optional [False]
                If True, make the network residual.
        """

        super().__init__(block_setting, create_linears=False)

        self.multiple_networks = block_setting.optional.get(
            'multiple_networks', True)
        if create_subchain:
            self.subchains, self.subchain_indices = self._create_subchains(
                block_setting)
        return

    def _create_subchains(
            self, block_setting,
            twice_input_nodes=False, square_weight=False, start_index=0):
        if self.multiple_networks:
            subchains = torch.nn.ModuleList([
                self._create_subsubchain(
                    block_setting,
                    twice_input_nodes=twice_input_nodes,
                    square_weight=square_weight, start_index=start_index)
                for _ in block_setting.support_input_indices])
            subchain_indices = range(
                len(block_setting.support_input_indices))
        else:
            subchains = torch.nn.ModuleList([
                self._create_subsubchain(
                    block_setting,
                    twice_input_nodes=twice_input_nodes,
                    square_weight=square_weight, start_index=start_index)])
            subchain_indices = [0] * len(
                block_setting.support_input_indices)

        return subchains, subchain_indices

    def _create_subsubchain(
            self, block_setting,
            twice_input_nodes=False, square_weight=False, start_index=0):
        nodes = block_setting.nodes[start_index:]
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
            h_res = torch.sum(torch.stack([
                self._forward_single_core(x, self.subchain_indices[i], support)
                for i, support in enumerate(supports)]), dim=0)
            return self.activations[-1](h_res + self.shortcut(x))
        else:
            return torch.sum(torch.stack([
                self._forward_single_core(x, self.subchain_indices[i], support)
                for i, support in enumerate(supports)]), dim=0)

    def _forward_single_core(self, x, subchain_index, support):
        raise NotImplementedError


import torch


def identity(x):
    return x


def max_pool(x):
    return torch.max(x, dim=-2, keepdim=True)[0]


def mean(x):
    return torch.mean(x, dim=-2, keepdim=True)


def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))


DICT_ACTIVATIONS = {
    'identity': identity,
    'relu': torch.relu,
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
    'max_pool': max_pool,
    'max': max_pool,
    'mean': mean,
    'mish': mish,
    'softplus': torch.nn.functional.softplus,
}


class AbstractMLP(torch.nn.Module):

    def __init__(self, block_setting, last_identity=False):
        """Initialize the NN.

        Parameters
        -----------
        block_setting: siml.setting.BlockSetting
            BlockSetting object.
        last_identity: bool
            If True, set the last activation identity whatever the
            block_setting (default: False).
        """
        super().__init__()

        nodes = block_setting.nodes
        try:
            self.linears = torch.nn.ModuleList([
                torch.nn.Linear(n1, n2)
                for n1, n2 in zip(nodes[:-1], nodes[1:])])
        except RuntimeError:
            raise ValueError(f"Cannot cretate linear for {block_setting}")
        self.activations = [
            DICT_ACTIVATIONS[activation]
            for activation in block_setting.activations]
        if last_identity:
            self.activations[-1] = DICT_ACTIVATIONS['identity']
        self.dropout_ratios = [
            dropout_ratio for dropout_ratio in block_setting.dropouts]

    def forward(self, x, support=None):
        raise NotImplementedError


class AbstractGCN(torch.nn.Module):

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

        super().__init__()
        self.residual = residual

        self.multiple_networks = block_setting.optional.get(
            'multiple_networks', True)
        if create_subchain:
            self.subchains, self.subchain_indices = self._create_subchains(
                block_setting)

        self.dropout_ratios = [
            dropout_ratio for dropout_ratio in block_setting.dropouts]

        if self.residual:
            activations = [
                DICT_ACTIVATIONS[activation]
                for activation in block_setting.activations]
            self.activations = activations[:-1] \
                + [DICT_ACTIVATIONS['identity']] \
                + [activations[-1]]
            nodes = block_setting.nodes
            if nodes[0] == nodes[-1]:
                self.shortcut = identity
            else:
                self.shortcut = torch.nn.Linear(nodes[0], nodes[-1])
        else:
            self.activations = [
                DICT_ACTIVATIONS[activation]
                for activation in block_setting.activations]

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

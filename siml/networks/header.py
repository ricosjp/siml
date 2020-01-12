
import torch
import torch.nn.functional as functional


def identity(x):
    return x


def max_pool(x):
    return torch.max(x, dim=-2, keepdim=True)[0]


def mean(x):
    return torch.mean(x, dim=-2, keepdim=True)


DICT_ACTIVATIONS = {
    'identity': identity,
    'relu': functional.relu,
    'sigmoid': functional.sigmoid,
    'tanh': functional.tanh,
    'max_pool': max_pool,
    'mean': mean,
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
        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(n1, n2) for n1, n2 in zip(nodes[:-1], nodes[1:])])
        self.activations = [
            DICT_ACTIVATIONS[activation]
            for activation in block_setting.activations]
        if last_identity:
            self.activations[-1] = DICT_ACTIVATIONS['identity']
        self.dropout_ratios = [
            dropout_ratio for dropout_ratio in block_setting.dropouts]
        self.input_selection = block_setting.input_selection

    def __call__(self, x, support=None):
        raise NotImplementedError


class AbstractGCN(torch.nn.Module):

    def __init__(self, block_setting, *, create_subchain=True):
        """Initialize the NN.

        Parameters
        -----------
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
            create_subchain: bool, optional [True]
                If True, create subchain to be trained.
        """

        super().__init__()

        nodes = block_setting.nodes
        self.subchains = torch.nn.ModuleList([
            torch.nn.Linear(n1, n2)
            for n1, n2 in zip(nodes[:-1], nodes[1:])])

        self.activations = [
            DICT_ACTIVATIONS[activation]
            for activation in block_setting.activations]
        self.dropout_ratios = [
            dropout_ratio for dropout_ratio in block_setting.dropouts]
        self.support_input_index = block_setting.support_input_index
        self.input_selection = block_setting.input_selection

    def __call__(self, x, supports):
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
        hs = torch.stack([
            self._call_single(
                x_[:, self.input_selection],
                supports_[self.support_input_index])
            for x_, supports_ in zip(x, supports)])
        return hs

    def _call_single(self, x, support):
        raise NotImplementedError

import chainer as ch
import numpy as np

from . import adjustable_mlp
from . import distributor
from . import gcn
from . import identity
from . import mlp


class BlockInformation():

    def __init__(self, block, use_support=False):
        self.block = block
        self.use_support = use_support


class Network(ch.Chain):

    DICT_BLOCKS = {
        'identity': BlockInformation(identity.Identity),
        'mlp': BlockInformation(mlp.MLP),
        'adjustable_mlp': BlockInformation(adjustable_mlp.AdjustableMLP),
        'adjustable_brick_mlp': BlockInformation(
            adjustable_mlp.AdjustableBrickMLP),
        'gcn': BlockInformation(gcn.GCN, use_support=True),
        'res_gcn': BlockInformation(gcn.ResGCN, use_support=True),
        'distributor': BlockInformation(distributor.Distributor),
    }

    def __init__(self, model_setting, trainer_setting):
        super().__init__()
        self.model_setting = model_setting
        block_informations = [
            self.DICT_BLOCKS[block.type] for block
            in self.model_setting.blocks]
        with self.init_scope():
            self.chains = ch.ChainList(*[
                block_information.block(block_setting)
                for block_information, block_setting
                in zip(block_informations, self.model_setting.blocks)])
        self.call_graph = self._create_call_graph()
        self.use_support = trainer_setting.support_inputs is not None
        self.support_indices = [
            block_setting.support_input_index
            if block_information.use_support else 0
            for block_information, block_setting
            in zip(block_informations, self.model_setting.blocks)]

        if self.use_support:
            self._call_core = self._call_with_support
        else:
            self._call_core = self._call_without_support

    def _create_call_graph(self):
        list_destinations = [
            block.destinations for block in self.model_setting.blocks]
        names = np.array(
            [block.name for block in self.model_setting.blocks] + ['Output'])
        list_destination_indices = [
            self._find_destination_indices(destinations, names)
            for destinations in list_destinations]
        call_graph = [[] for _ in range(len(self.chains) + 1)]
        for i, destination_indices in enumerate(list_destination_indices):
            for destination_index in destination_indices:
                call_graph[destination_index].append(i)
        return call_graph

    def _find_destination_indices(self, destinations, names):
        locations = [
            np.where(destination == names)[0] for destination in destinations]
        if not np.all(
                np.array([len(location) for location in locations]) == 1):
            raise ValueError('Destination name is not unique')
        return np.ravel(locations)

    def __call__(self, x, supports=None):
        return self._call_core(x, supports)

    def _call_with_support(self, x, supports):
        hiddens = [None] * len(self.chains)

        hiddens[0] = self.chains[0](x, supports)
        for i in range(1, len(hiddens)):
            inputs = [hiddens[input_node] for input_node in self.call_graph[i]]
            hiddens[i] = self.chains[i](*inputs, supports=supports)
        return hiddens[-1]

    def _call_without_support(self, x, supports=None):
        hiddens = [None] * len(self.chains)
        hiddens[0] = self.chains[0](x)
        for i in range(1, len(hiddens)):
            for input_node in self.call_graph[i]:
                hiddens[i] = self.chains[i](hiddens[input_node])
        return hiddens[-1]

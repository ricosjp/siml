import warnings

import numpy as np
import torch

from . import adjustable_mlp
from . import deepsets
from . import gcn
from . import identity
from . import mlp
from . import nri
from . import reducer


class BlockInformation():

    def __init__(self, block, use_support=False):
        self.block = block
        self.use_support = use_support


class Network(torch.nn.Module):

    DICT_BLOCKS = {
        'identity': BlockInformation(identity.Identity),
        'mlp': BlockInformation(mlp.MLP),
        'adjustable_mlp': BlockInformation(adjustable_mlp.AdjustableMLP),
        'gcn': BlockInformation(gcn.GCN, use_support=True),
        'res_gcn': BlockInformation(gcn.ResGCN, use_support=True),
        'reducer': BlockInformation(reducer.Reducer),
        'distributor': BlockInformation(reducer.Reducer),  # For backward compatibility  # NOQA
        'deepsets': BlockInformation(deepsets.DeepSets),
        'nri': BlockInformation(nri.NRI, use_support=True),
    }

    def __init__(self, model_setting, trainer_setting):
        super().__init__()
        self.model_setting = model_setting

        for block in self.model_setting.blocks:
            if 'distributor' == block.type:
                warnings.warn(
                    'distributor type is deprecated. Use reducer',
                    DeprecationWarning)

        block_informations = [
            self.DICT_BLOCKS[block.type] for block
            in self.model_setting.blocks]

        self.use_support_informations = [
            block_information.use_support
            for block_information in block_informations]
        self.use_support = np.any(self.use_support_informations)

        self.chains = torch.nn.ModuleList([
            block_information.block(block_setting)
            for block_information, block_setting
            in zip(block_informations, self.model_setting.blocks)])
        self.call_graph = self._create_call_graph()
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

    def __call__(self, x):
        return self._call_core(x)

    def _call_with_support(self, x_):
        x = x_['x']
        supports = x_['supports']
        hiddens = [None] * len(self.chains)

        hiddens[0] = self.chains[0](x, supports)
        for i in range(1, len(hiddens)):
            inputs = [hiddens[input_node] for input_node in self.call_graph[i]]
            if self.use_support_informations[i]:
                hiddens[i] = self.chains[i](*inputs, supports=supports)
            else:
                hiddens[i] = self.chains[i](*inputs)
        return hiddens[-1]

    def _call_without_support(self, x_):
        x = x_['x']
        hiddens = [None] * len(self.chains)
        hiddens[0] = self.chains[0](x)
        for i in range(1, len(hiddens)):
            inputs = [hiddens[input_node] for input_node in self.call_graph[i]]
            hiddens[i] = self.chains[i](*inputs)
        return hiddens[-1]

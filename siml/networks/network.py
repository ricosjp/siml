import warnings

import numpy as np
import torch

from .. import datasets
from . import adjustable_mlp
from . import deepsets
from . import gcn
from . import identity
from . import lstm
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
        'distributor': BlockInformation(
            reducer.Reducer),  # For backward compatibility
        'deepsets': BlockInformation(deepsets.DeepSets),
        'nri': BlockInformation(nri.NRI, use_support=True),
        'lstm': BlockInformation(lstm.LSTM),
        'res_lstm': BlockInformation(lstm.ResLSTM),
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
        self.devices = [
            block_setting.device
            for block_setting in self.model_setting.blocks]

        self.chains = torch.nn.ModuleList([
            block_information.block(block_setting).to(block_setting.device)
            for block_information, block_setting
            in zip(block_informations, self.model_setting.blocks)])
        self.call_graph = self._create_call_graph()
        self.support_indices = [
            block_setting.support_input_index
            if block_information.use_support else 0
            for block_information, block_setting
            in zip(block_informations, self.model_setting.blocks)]

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

    def forward(self, x):
        if self.use_support:
            return self._forward_with_support(x)
        else:
            return self._forward_without_support(x)

    def _forward_with_support(self, x_):
        x = x_['x']
        # Due to lack of support of sparse matrix of scatter in DataParallel
        # and coo_matrix, convert sparse in the forward
        block_settings = self.model_setting.blocks
        supports = datasets.convert_sparse_tensor(
            x_['supports'], device=x.device)
        hiddens = [None] * len(self.chains)

        hiddens[0] = self.chains[0](
            x[..., block_settings[0].input_selection],
            supports[:, block_settings[0].support_input_indices])
        for i in range(1, len(hiddens)):
            device = self.devices[i]
            inputs = [
                hiddens[input_node][
                    ..., block_settings[i].input_selection].to(device)
                for input_node in self.call_graph[i]]
            if self.use_support_informations[i]:
                selected_supports = [
                    [s.to(device) for s in sp] for sp
                    in supports[:, block_settings[i].support_input_indices]]
                hiddens[i] = self.chains[i](
                    *inputs, supports=selected_supports)
            else:
                hiddens[i] = self.chains[i](*inputs)
        return hiddens[-1]

    def _forward_without_support(self, x_):
        block_settings = self.model_setting.blocks
        x = x_['x']
        hiddens = [None] * len(self.chains)
        hiddens[0] = self.chains[0](x[..., block_settings[0].input_selection])
        for i in range(1, len(hiddens)):
            device = self.devices[i]
            inputs = [
                hiddens[input_node][
                    ..., block_settings[i].input_selection].to(device)
                for input_node in self.call_graph[i]]
            hiddens[i] = self.chains[i](*inputs)
        return hiddens[-1]

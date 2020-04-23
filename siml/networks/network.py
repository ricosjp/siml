# from io import BytesIO
import warnings

import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import networkx as nx
import numpy as np
import torch

from .. import datasets
from .. import setting
from . import activation
from . import concatenator
from . import deepsets
from . import gcn
from . import grad_gcn
from . import identity
from . import integration
from . import laplace_net
from . import lstm
from . import mlp
from . import nri
from . import reducer
from . import tcn
from . import time_norm


class BlockInformation():

    def __init__(self, block, use_support=False, trainable=True):
        self.block = block
        self.use_support = use_support
        self.trainable = trainable


class Network(torch.nn.Module):

    DICT_BLOCKS = {
        # Layers without weights
        'activation': BlockInformation(activation.Activation, trainable=False),
        'concatenator': BlockInformation(
            concatenator.Concatenator, trainable=False),
        'distributor': BlockInformation(
            reducer.Reducer, trainable=False),  # For backward compatibility
        'identity': BlockInformation(identity.Identity, trainable=False),
        'integration': BlockInformation(
            integration.Integration, trainable=False),
        'reducer': BlockInformation(reducer.Reducer, trainable=False),
        'time_norm': BlockInformation(time_norm.TimeNorm, trainable=False),

        # Layers with weights
        'adjustable_mlp': BlockInformation(mlp.MLP),
        'deepsets': BlockInformation(deepsets.DeepSets),
        'gcn': BlockInformation(gcn.GCN, use_support=True),
        'grad_gcn': BlockInformation(grad_gcn.GradGCN, use_support=True),
        'laplace_net': BlockInformation(
            laplace_net.LaplaceNet, use_support=True),
        'lstm': BlockInformation(lstm.LSTM),
        'mlp': BlockInformation(mlp.MLP),
        'nri': BlockInformation(nri.NRI, use_support=True),
        'tcn': BlockInformation(tcn.TCN),
    }
    INPUT_LAYER_NAME = 'Input'
    OUTPUT_LAYER_NAME = 'Output'

    def __init__(self, model_setting, trainer_setting):
        super().__init__()
        self.model_setting = model_setting
        self.trainer_setting = trainer_setting

        for block in self.model_setting.blocks:
            if 'distributor' == block.type:
                warnings.warn(
                    'distributor type is deprecated. Use reducer',
                    DeprecationWarning)

        self.dict_block_setting = {
            block_setting.name: block_setting
            for block_setting in self.model_setting.blocks}

        self.call_graph = self._create_call_graph()
        self.sorted_graph_nodes = list(nx.topological_sort(self.call_graph))

        self._update_dict_block_setting()
        self.dict_block_information = {
            block_name: self.DICT_BLOCKS[block_setting.type]
            for block_name, block_setting in self.dict_block_setting.items()}
        self.dict_block = self._create_dict_block()

        self.use_support = np.any([
            block_information.use_support
            for block_information in self.dict_block_information.values()])

        return

    def _create_call_graph(self):
        call_graph = nx.DiGraph()
        block_names = [
            block_setting.name for block_setting
            in self.model_setting.blocks] + [
                self.INPUT_LAYER_NAME, self.OUTPUT_LAYER_NAME]
        for block_setting in self.model_setting.blocks:
            if block_setting.name == self.INPUT_LAYER_NAME:
                raise ValueError(
                    f"Do not use block names: {self.INPUT_LAYER_NAME}")
            if block_setting.name == self.OUTPUT_LAYER_NAME:
                raise ValueError(
                    f"Do not use block names: {self.OUTPUT_LAYER_NAME}")

            for destination in block_setting.destinations:
                if destination not in block_names:
                    raise ValueError(f"{destination} does not exist")
                call_graph.add_edge(block_setting.name, destination)
            if block_setting.is_first:
                call_graph.add_edge(self.INPUT_LAYER_NAME, block_setting.name)
            if block_setting.is_last:
                call_graph.add_edge(block_setting.name, self.OUTPUT_LAYER_NAME)

        # Validate call graph
        if not nx.is_directed_acyclic_graph(call_graph):
            cycle = nx.find_cycle(call_graph)
            raise ValueError(
                f"Cycle found in the network: {cycle}")
        for graph_node in call_graph.nodes():
            predecessors = tuple(call_graph.predecessors(graph_node))
            successors = tuple(call_graph.successors(graph_node))
            if len(predecessors) == 0 and graph_node != self.INPUT_LAYER_NAME:
                raise ValueError(f"{graph_node} has no predecessors")
            if len(successors) == 0 and graph_node != self.OUTPUT_LAYER_NAME:
                raise ValueError(f"{graph_node} has no successors")
        return call_graph

    def _update_dict_block_setting(self):
        self.dict_block_setting.update({
            self.INPUT_LAYER_NAME: setting.BlockSetting(
                name=self.INPUT_LAYER_NAME, type='identity'),
            self.OUTPUT_LAYER_NAME: setting.BlockSetting(
                name=self.OUTPUT_LAYER_NAME, type='identity')})

        for graph_node in self.sorted_graph_nodes:
            predecessors = tuple(self.call_graph.predecessors(graph_node))
            block_setting = self.dict_block_setting[graph_node]
            block_type = block_setting.type

            if graph_node == self.INPUT_LAYER_NAME:
                first_node = self.trainer_setting.input_length
                last_node = self.trainer_setting.input_length

            elif graph_node == self.OUTPUT_LAYER_NAME:
                first_node = self.trainer_setting.output_length
                last_node = self.trainer_setting.output_length

            elif block_type == 'concatenator':
                max_first_node = np.sum([
                    self.dict_block_setting[predecessor].nodes[-1]
                    for predecessor in predecessors])
                first_node = len(np.arange(max_first_node)[
                    block_setting.input_selection])
                last_node = first_node

            elif block_type == 'reducer':
                max_first_node = np.sum([
                    self.dict_block_setting[predecessor].nodes[-1]
                    for predecessor in predecessors])
                first_node = len(np.arange(max_first_node)[
                    block_setting.input_selection])
                last_node = np.max([
                    self.dict_block_setting[predecessor].nodes[-1]
                    for predecessor in predecessors])

            elif block_type == 'integration':
                max_first_node = self.dict_block_setting[
                    predecessors[0]].nodes[-1]
                first_node = len(np.arange(max_first_node)[
                    block_setting.input_selection])
                last_node = first_node - 1

            elif self.DICT_BLOCKS[block_type].trainable:
                # Trainable layers
                if len(predecessors) != 1:
                    raise ValueError(
                        f"{graph_node} has {len(predecessors)} "
                        f"predecessors: {predecessors}")
                max_first_node = self.dict_block_setting[
                    tuple(predecessors)[0]].nodes[-1]
                first_node = len(np.arange(max_first_node)[
                    block_setting.input_selection])
                last_node = self.trainer_setting.output_length

            else:
                # Non trainable other layers
                if len(predecessors) != 1:
                    raise ValueError(
                        f"{graph_node} has {len(predecessors)} "
                        f"predecessors: {predecessors}")
                max_first_node = self.dict_block_setting[
                    predecessors[0]].nodes[-1]
                first_node = len(np.arange(max_first_node)[
                    block_setting.input_selection])
                last_node = first_node

            if self.dict_block_setting[graph_node].nodes[0] == -1:
                self.dict_block_setting[graph_node].nodes[0] = int(first_node)

            if self.DICT_BLOCKS[block_type].trainable:
                if self.dict_block_setting[graph_node].nodes[-1] == -1:
                    self.dict_block_setting[graph_node].nodes[-1] = int(
                        last_node)
            else:
                if self.dict_block_setting[graph_node].nodes[-1] == -1:
                    self.dict_block_setting[graph_node].nodes[-1] = int(
                        last_node)

        return

    def _create_dict_block(self):
        dict_block = torch.nn.ModuleDict({
            block_name:
            self.dict_block_information[block_name].block(block_setting).to(
                block_setting.device)
            for block_name, block_setting in self.dict_block_setting.items()})
        return dict_block

    def forward(self, x_):
        x = x_['x']

        # Due to lack of support of sparse matrix of scatter in DataParallel
        # and coo_matrix, convert sparse in the forward
        if self.use_support:
            supports = datasets.convert_sparse_tensor(
                x_['supports'], device=x.device)
        dict_hidden = {
            block_name: None for block_name in self.call_graph.nodes}

        for graph_node in self.sorted_graph_nodes:
            block_setting = self.dict_block_setting[graph_node]
            if graph_node == self.INPUT_LAYER_NAME:
                dict_hidden[graph_node] = x
            else:
                device = block_setting.device
                inputs = [
                    dict_hidden[predecessor][
                        ..., block_setting.input_selection].to(device)
                    for predecessor
                    in self.call_graph.predecessors(graph_node)]

                if self.dict_block_information[graph_node].use_support:
                    selected_supports = [
                        [s.to(device) for s in sp] for sp
                        in supports[:, block_setting.support_input_indices]]
                    dict_hidden[graph_node] = self.dict_block[graph_node](
                        *inputs, supports=selected_supports)
                else:
                    dict_hidden[graph_node] = self.dict_block[graph_node](
                        *inputs)

        return dict_hidden[self.OUTPUT_LAYER_NAME]

    def draw(self, output_file_name):
        figure = plt.figure(dpi=1000)
        mapping = {
            graph_node:
            f"{graph_node}\n"
            f"{self.dict_block_setting[graph_node].type}\n"
            f"{self.dict_block_setting[graph_node].nodes}"
            for graph_node in self.sorted_graph_nodes}
        d = nx.drawing.nx_pydot.to_pydot(nx.relabel.relabel_nodes(
            self.call_graph, mapping=mapping, copy=True))
        d.write_pdf(output_file_name)
        # pdf_str = d.create_pdf()
        # sio = BytesIO()
        # sio.write(pdf_str)
        # sio.seek(0)
        # img = mpimg.imread(sio)
        # plt.axis('off')
        # plt.imshow(img)
        # plt.savefig(output_file_name)
        plt.close(figure)

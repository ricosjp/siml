import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from .. import config
from .. import setting


class Network(torch.nn.Module):

    dict_block_class = {}

    def __init__(self, model_setting, trainer_setting):
        super().__init__()
        self.model_setting = model_setting
        self.trainer_setting = trainer_setting
        self.y_dict_mode = isinstance(self.trainer_setting.outputs, dict)

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
            block_name: self.dict_block_class[block_setting.type]
            for block_name, block_setting in self.dict_block_setting.items()}
        self.dict_block = self._create_dict_block()

        self.use_support = np.any([
            block_information.uses_support()
            for block_information in self.dict_block_information.values()])
        self.merge_sparses = False
        if self.merge_sparses:
            print('Sparse matrices are merged for IsoGCN')

        return

    def _create_call_graph(self):
        call_graph = nx.DiGraph()
        block_names = [
            block_setting.name for block_setting
            in self.model_setting.blocks] + [
                config.INPUT_LAYER_NAME, config.OUTPUT_LAYER_NAME]
        for block_setting in self.model_setting.blocks:
            if block_setting.name == config.INPUT_LAYER_NAME:
                raise ValueError(
                    f"Do not use block names: {config.INPUT_LAYER_NAME}")
            if block_setting.name == config.OUTPUT_LAYER_NAME:
                raise ValueError(
                    f"Do not use block names: {config.OUTPUT_LAYER_NAME}")

            for destination in block_setting.destinations:
                if destination not in block_names:
                    raise ValueError(f"{destination} does not exist")
                call_graph.add_edge(block_setting.name, destination)
            if block_setting.is_first:
                call_graph.add_edge(
                    config.INPUT_LAYER_NAME, block_setting.name)
            if block_setting.is_last:
                call_graph.add_edge(
                    block_setting.name, config.OUTPUT_LAYER_NAME)

        # Validate call graph
        if not nx.is_directed_acyclic_graph(call_graph):
            cycle = nx.find_cycle(call_graph)
            raise ValueError(
                f"Cycle found in the network: {cycle}")
        for graph_node in call_graph.nodes():
            predecessors = tuple(call_graph.predecessors(graph_node))
            successors = tuple(call_graph.successors(graph_node))
            if len(predecessors) == 0 \
                    and graph_node != config.INPUT_LAYER_NAME:
                raise ValueError(f"{graph_node} has no predecessors")
            if len(successors) == 0 \
                    and graph_node != config.OUTPUT_LAYER_NAME:
                raise ValueError(f"{graph_node} has no successors")
        return call_graph

    def _update_dict_block_setting(self):
        self.dict_block_setting.update({
            config.INPUT_LAYER_NAME: setting.BlockSetting(
                name=config.INPUT_LAYER_NAME, type='identity'),
            config.OUTPUT_LAYER_NAME: setting.BlockSetting(
                name=config.OUTPUT_LAYER_NAME, type='identity')})

        for graph_node in self.sorted_graph_nodes:
            block_setting = self.dict_block_setting[graph_node]
            predecessors = tuple(self.call_graph.predecessors(graph_node))
            block_type = block_setting.type
            block_class = self.dict_block_class[block_type]

            input_node, output_node = block_class.get_n_nodes(
                block_setting, predecessors, self.dict_block_setting,
                self.trainer_setting.input_length,
                self.trainer_setting.output_length)
            block_setting.nodes[0] = input_node
            block_setting.nodes[-1] = output_node

        # for graph_node in self.sorted_graph_nodes:
        #     block_setting = self.dict_block_setting[graph_node]
        #     predecessors = tuple(self.call_graph.predecessors(graph_node))
        #     block_type = block_setting.type
        #
        #     if graph_node == config.INPUT_LAYER_NAME:
        #         first_node = self.trainer_setting.input_length
        #         last_node = self.trainer_setting.input_length
        #
        #     elif graph_node == config.OUTPUT_LAYER_NAME:
        #         first_node = self.trainer_setting.output_length
        #         last_node = self.trainer_setting.output_length
        #
        #     elif block_type == 'array2symmat':
        #         first_node == 6
        #         last_node = 1
        #
        #     elif block_type == 'symmat2array':
        #         max_first_node = np.sum([
        #             self.dict_block_setting[predecessor].nodes[-1]
        #             for predecessor in predecessors])
        #         first_node = max(len(np.arange(max_first_node)[
        #             block_setting.input_selection]), 1)
        #         last_node = first_node * 6
        #
        #     # elif block_type == 'concatenator':
        #     #     max_first_node = np.sum([
        #     #         self.dict_block_setting[predecessor].nodes[-1]
        #     #         for predecessor in predecessors])
        #     #     first_node = len(np.arange(max_first_node)[
        #     #         block_setting.input_selection])
        #     #     last_node = first_node
        #
        #     elif block_type in ['reducer', 'contraction']:
        #         max_first_node = np.sum([
        #             self.dict_block_setting[predecessor].nodes[-1]
        #             for predecessor in predecessors])
        #         first_node = len(np.arange(max_first_node)[
        #             block_setting.input_selection])
        #         last_node = np.max([
        #             self.dict_block_setting[predecessor].nodes[-1]
        #             for predecessor in predecessors])
        #
        #     elif block_type == 'reshape':
        #         max_first_node = np.sum([
        #             self.dict_block_setting[predecessor].nodes[-1]
        #             for predecessor in predecessors])
        #         first_node = len(np.arange(max_first_node)[
        #             block_setting.input_selection])
        #         last_node = block_setting.optional['new_shape'][1]
        #
        #     elif block_type == 'integration':
        #         max_first_node = self.dict_block_setting[
        #             predecessors[0]].nodes[-1]
        #         first_node = len(np.arange(max_first_node)[
        #             block_setting.input_selection])
        #         last_node = first_node - 1
        #
        #     else:
        #         if block_setting.is_first:
        #             if isinstance(self.trainer_setting.input_length, dict):
        #                 input_keys = block_setting.input_keys
        #                 if input_keys is None:
        #                     raise ValueError(
        #                         'Input is dict. Plese specify input_keys to '
        #                         f"the first nodes: {block_setting}")
        #                 input_length = self.trainer_setting.input_length
        #                 max_first_node = int(
        #                     np.sum([
        #                         input_length[input_key] for input_key
        #                         in input_keys]))
        #             else:
        #                 max_first_node = self.trainer_setting.input_length
        #         else:
        #             max_first_node = self.dict_block_setting[
        #                 tuple(predecessors)[0]].nodes[-1]
        #         first_node = len(np.arange(max_first_node)[
        #             block_setting.input_selection])
        #
        #         if self.dict_block_class[block_type].trainable:
        #             last_node = self.trainer_setting.output_length
        #         else:
        #             last_node = first_node
        #
        #     if graph_node not in [
        #             config.INPUT_LAYER_NAME, config.OUTPUT_LAYER_NAME] \
        #             and block_setting.nodes[0] == -1:
        #         block_setting.nodes[0] = int(first_node)
        #
        #     if graph_node not in [
        #             config.INPUT_LAYER_NAME, config.OUTPUT_LAYER_NAME] \
        #             and block_setting.nodes[-1] == -1:
        #         output_key = block_setting.output_key
        #         if output_key is None:
        #             if isinstance(last_node, dict):
        #                 raise ValueError(
        #                     'Output is dict. Plese specify output_key to the '
        #                     f"last nodes: {block_setting}")
        #             block_setting.nodes[-1] = int(
        #                 last_node)
        #         else:
        #             if block_setting.is_last:
        #                 output_length = self.trainer_setting.output_length
        #                 block_setting.nodes[-1] = int(
        #                     output_length[output_key])
        #             else:
        #                 raise ValueError(
        #                     'Cannot put output_key when is_last is False: '
        #                     f"{block_setting}")

        return

    def _create_dict_block(self):
        dict_block = torch.nn.ModuleDict({
            block_name:
            self.dict_block_information[block_name](block_setting).to(
                block_setting.device)
            for block_name, block_setting in self.dict_block_setting.items()})
        return dict_block

    def forward(self, x_):
        x = x_['x']
        supports = x_.get('supports', None)
        original_shapes = x_.get('original_shapes', None)

        dict_hidden = {
            block_name: None for block_name in self.call_graph.nodes}

        for graph_node in self.sorted_graph_nodes:
            block_setting = self.dict_block_setting[graph_node]
            if graph_node == config.INPUT_LAYER_NAME:
                dict_hidden[graph_node] = x
            else:
                device = block_setting.device

                if block_setting.input_keys is None:
                    inputs = [
                        self._select_dimension(
                            dict_hidden[predecessor],
                            block_setting.input_selection, device)
                        for predecessor
                        in self.call_graph.predecessors(graph_node)]
                else:
                    inputs = [
                        torch.cat([
                            dict_hidden[predecessor][input_key][
                                ..., block_setting.input_selection].to(device)
                            for input_key in block_setting.input_keys], dim=-1)
                        for predecessor
                        in self.call_graph.predecessors(graph_node)]

                if self.dict_block_information[graph_node].uses_support():
                    if self.merge_sparses:
                        # NOTE: support_input_indices are ignored
                        selected_supports = supports
                    else:
                        if isinstance(supports[0], list):
                            selected_supports = [
                                [s.to(device) for s in sp] for sp
                                in supports[
                                    :, block_setting.support_input_indices]]
                        else:
                            selected_supports = [
                                supports[i].to(device) for i
                                in block_setting.support_input_indices]

                    hidden = self.dict_block[graph_node](
                        *inputs, supports=selected_supports,
                        original_shapes=original_shapes)
                else:
                    hidden = self.dict_block[graph_node](
                        *inputs, original_shapes=original_shapes)

                if block_setting.coeff is not None:
                    hidden = hidden * block_setting.coeff
                if block_setting.output_key is None:
                    dict_hidden[graph_node] = hidden
                else:
                    dict_hidden[graph_node] = {
                        block_setting.output_key: hidden}

        if self.y_dict_mode:
            return_dict = {}
            if isinstance(dict_hidden[config.OUTPUT_LAYER_NAME], dict):
                return_dict.update(dict_hidden[config.OUTPUT_LAYER_NAME])
            else:
                for h in dict_hidden[config.OUTPUT_LAYER_NAME]:
                    return_dict.update(h)
            return return_dict
        else:
            return dict_hidden[config.OUTPUT_LAYER_NAME]

    def _select_dimension(self, x, input_selection, device):
        if isinstance(x, dict):
            if input_selection != slice(0, None, 1):
                raise ValueError(
                    'Cannot set input_selection after dict_output')
            return {key: value.to(device) for key, value in x.items()}
        else:
            if input_selection == slice(0, None, 1):
                return x
            else:
                return x[..., input_selection].to(device)

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
        if self.trainer_setting.figure_format == 'pdf':
            d.write_pdf(output_file_name)
        elif self.trainer_setting.figure_format == 'png':
            d.write_png(output_file_name)
        plt.close(figure)
        return


def add_block(block):
    """Add block definition to siml.

    Parameters
    ----------
    block: siml.network.SimlModule-like
        User defined block.
    """
    Network.dict_block_class.update({block.get_name(): block})
    return

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
        self.y_dict_mode = isinstance(
            self.trainer_setting.outputs.variables, dict)

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

        self.loss_blocks = [
            key for key, value in self.dict_block_setting.items()
            if len(value.losses) > 0]
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
            if block_type not in self.dict_block_class:
                raise ValueError(
                    f"{block_type} invalid for: {block_setting}\n"
                    f"Not in: {self.dict_block_class}")
            block_class = self.dict_block_class[block_type]

            input_node, output_node = block_class.get_n_nodes(
                block_setting, predecessors, self.dict_block_setting,
                self.trainer_setting.input_length,
                self.trainer_setting.output_length,
                self.model_setting)
            block_setting.nodes[0] = input_node
            block_setting.nodes[-1] = output_node

        return

    def _create_dict_block(self):
        dict_block = torch.nn.ModuleDict({
            block_name:
            self.dict_block_information[block_name](block_setting).to(
                block_setting.device)
            for block_name, block_setting in self.dict_block_setting.items()
            if (block_setting.reference_block_name is None) and
            (block_setting.type != 'group')
        })

        # Update dict_block for blocks depending on other blocks
        dict_block.update(torch.nn.ModuleDict({
            block_name:
            self.dict_block_information[block_name](
                block_setting,
                reference_block=dict_block[
                    block_setting.reference_block_name]
            ).to(block_setting.device)
            for block_name, block_setting in self.dict_block_setting.items()
            if block_setting.reference_block_name is not None
        }))

        # Update dict_block for blocks depending on other groups
        dict_block.update(torch.nn.ModuleDict({
            block_name:
            self.dict_block_information[block_name](
                block_setting,
                model_setting=self.model_setting)
            for block_name, block_setting in self.dict_block_setting.items()
            if block_setting.type == 'group'
        }))
        return dict_block

    def get_loss_keys(self):
        return [
            f"{loss_block}/{loss_name}"
            for loss_block in self.loss_blocks
            for loss_name
            in self.dict_block[loss_block].block_setting.loss_names]

    def get_losses(self):
        return {
            f"{loss_block}/{loss_setting['name']}":
            self.dict_block[loss_block].losses[loss_setting['name']]
            for loss_block in self.loss_blocks
            for loss_setting
            in self.dict_block[loss_block].block_setting.losses}

    def get_loss_coeffs(self):
        return {
            f"{loss_block}/{loss_setting['name']}": loss_setting['coeff']
            for loss_block in self.loss_blocks
            for loss_setting
            in self.dict_block[loss_block].block_setting.losses}

    def reset(self):
        for loss_block in self.loss_blocks:
            self.dict_block[loss_block].reset()
        return

    def forward(self, x_):
        x = x_['x']
        supports = x_.get('supports', None)
        original_shapes = x_.get('original_shapes', None)

        dict_hidden = {
            block_name: None for block_name in self.call_graph.nodes}

        for graph_node in self.sorted_graph_nodes:
            try:
                block_setting = self.dict_block_setting[graph_node]
                if graph_node == config.INPUT_LAYER_NAME:
                    dict_hidden[graph_node] = x
                else:
                    device = block_setting.device

                    if block_setting.type == 'group':
                        dict_predecessors = {
                            predecessor: dict_hidden[predecessor]
                            for predecessor
                            in self.call_graph.predecessors(graph_node)}
                        inputs = self.dict_block[graph_node].generate_inputs(
                            dict_predecessors)
                    elif block_setting.input_keys is None \
                            and block_setting.input_names is None:
                        inputs = [
                            self._select_dimension(
                                dict_hidden[predecessor],
                                block_setting.input_selection, device)
                            for predecessor
                            in self.call_graph.predecessors(graph_node)]
                    elif block_setting.input_keys is not None:
                        inputs = [
                            torch.cat(
                                [
                                    dict_hidden[predecessor][input_key][
                                        ..., block_setting.input_selection
                                    ].to(device)
                                    for input_key in block_setting.input_keys
                                ], dim=-1)
                            for predecessor
                            in self.call_graph.predecessors(graph_node)]

                    elif block_setting.input_names is not None:
                        if set(block_setting.input_names) != set(
                                self.call_graph.predecessors(graph_node)):
                            set_predecessors = set(
                                self.call_graph.predecessors(graph_node))
                            raise ValueError(
                                'input_names differs from the predecessors:\n'
                                f"{set(block_setting.input_names)}\n"
                                f"{set_predecessors}\n"
                                f"in: {block_setting}")
                        inputs = [
                            dict_hidden[input_name][
                                ..., block_setting.input_selection].to(device)
                            for input_name in block_setting.input_names]
                    else:
                        raise ValueError('Should not reach here')

                    if block_setting.type == 'group':
                        output = self.dict_block[graph_node](
                            inputs, supports=supports,
                            original_shapes=original_shapes)
                        hidden = output

                    elif self.dict_block_information[
                            graph_node].uses_support():
                        if self.merge_sparses:
                            raise ValueError(
                                'merge_sparses is no longer available')
                        else:
                            if isinstance(supports[0], list):
                                selected_supports = [
                                    [s.to(device) for s in sp] for sp
                                    in supports[
                                        :,
                                        block_setting.support_input_indices]]
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
                        if isinstance(hidden, dict):
                            hidden = {
                                k: v * block_setting.coeff
                                for k, v in hidden.items()}
                        else:
                            hidden = hidden * block_setting.coeff
                    if block_setting.output_key is not None:
                        dict_hidden[graph_node] = {
                            block_setting.output_key: hidden}
                    else:
                        dict_hidden[graph_node] = hidden
            except Exception as e:
                raise ValueError(
                    f"{e}\nError occured in: {block_setting}")

        if self.y_dict_mode:
            return_dict = {}
            if isinstance(dict_hidden[config.OUTPUT_LAYER_NAME], dict):
                return_dict.update(dict_hidden[config.OUTPUT_LAYER_NAME])
            elif isinstance(
                    dict_hidden[config.OUTPUT_LAYER_NAME], (list, tuple)):
                for h in dict_hidden[config.OUTPUT_LAYER_NAME]:
                    return_dict.update(h)
            else:
                if len(self.trainer_setting.outputs.variables.keys()) != 1:
                    raise ValueError(
                        'Invalid output setting: '
                        f"{self.trainer_setting.outputs.variables}")
                return_dict.update({
                    k: dict_hidden[config.OUTPUT_LAYER_NAME] for k
                    in self.trainer_setting.outputs.variables.keys()})
            return return_dict
        else:
            return dict_hidden[config.OUTPUT_LAYER_NAME]

    def clip_uniform_if_needed(
        self,
        clip_grad_value: float,
        clip_grad_norm: float
    ) -> None:
        if clip_grad_value is not None:
            torch.nn.utils.clip_grad_value_(
                self.parameters(), clip_grad_value)
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), clip_grad_norm)
        return

    def clip_if_needed(self):
        for graph_node in self.sorted_graph_nodes:
            block_setting = self.dict_block_setting[graph_node]

            if block_setting.type == 'group':
                self.dict_block[graph_node].group.clip_if_needed()

            if block_setting.clip_grad_value is not None:
                torch.nn.utils.clip_grad_value_(
                    self.dict_block[graph_node].parameters(),
                    block_setting.clip_grad_value)
            if block_setting.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.dict_block[graph_node].parameters(),
                    block_setting.clip_grad_norm)
        return

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

    def draw(self, output_directory, *, stem=None):
        if stem is None:
            stem = 'network'
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
            d.write_pdf(output_directory / f"{stem}.pdf")
        elif self.trainer_setting.figure_format == 'png':
            d.write_png(output_directory / f"{stem}.png")
        else:
            raise ValueError(
                f"Invalid figure format: {self.trainer_setting.figure_format}")
        plt.close(figure)

        # Draw groups if exist
        for key, block_setting in self.dict_block_setting.items():
            if block_setting.type != 'group':
                continue
            self.dict_block[key].group.draw(
                output_directory, stem=f"network_{block_setting.name}")

        return


def add_block(block):
    """Add block definition to siml.

    Parameters
    ----------
    block: siml.network.SimlModule-like
        User defined block.
    """
    name = block.get_name()
    if name in Network.dict_block_class:
        raise ValueError(f"Block name {name} already exists.")
    Network.dict_block_class.update({name: block})
    return

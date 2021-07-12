import abc

import numpy as np
import torch

from .. import config
from . import activations as acts


class SimlModule(torch.nn.Module, metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def get_name():
        """Abstract method to be overridden by the subclass.
        It shoud return str indicating its name.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def accepts_multiple_inputs():
        """Abstract method to be overridden by the subclass.
        It shoud return bool indicating if it accepts multiple inputs.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def is_trainable():
        """Abstract method to be overridden by the subclass.
        It shoud return bool indicating if it is trainable.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def uses_support():
        """Abstract method to be overridden by the subclass.
        It shoud return bool indicating if it uses support inputs
        (sparse matrices).
        """
        pass

    @classmethod
    def get_n_nodes(
            cls, block_setting, predecessors, dict_block_setting,
            input_length, output_length):
        """Get the number of input and output nodes.

        Parameters
        ----------
        block_setting: siml.setting.BlockSetting
            BlockSetting object of the block.
        predecessors: tuple[str]
            List of predecessor names.
        dict_block_setting: dict[siml.setting.BlockSetting]
            Dict of all BlockSetting objects.
        input_length: int or dict[int]
            Input length.
        output_length: int or dict[int]
            Output length.

        Returns
        -------
        input_node: int
            The number of the input nodes.
        output_node: int
            The number of the output nodes.
        """
        if block_setting.name == config.INPUT_LAYER_NAME:
            return -1, -1
        if block_setting.name == config.OUTPUT_LAYER_NAME:
            return -1, -1

        if not (cls.accepts_multiple_inputs() or len(predecessors) == 1):
            raise ValueError(
                f"{block_setting.name} has {len(predecessors)} "
                f"predecessors: {predecessors}")

        if block_setting.is_first:
            if isinstance(input_length, dict):
                input_keys = block_setting.input_keys
                if input_keys is None:
                    raise ValueError(
                        'Input is dict. Plese specify input_keys to '
                        f"the first nodes: {block_setting}")
                max_input_node = int(
                    np.sum([
                        input_length[input_key] for input_key
                        in input_keys]))
            else:
                max_input_node = input_length
        else:
            max_input_node = cls._get_n_input_node(
                block_setting, predecessors, dict_block_setting, input_length)
        if block_setting.nodes[0] == -1:
            input_node = len(np.arange(max_input_node)[
                block_setting.input_selection])
        else:
            input_node = block_setting.nodes[0]

        candidate_output_node = cls._get_n_output_node(
            input_node, block_setting, predecessors,
            dict_block_setting, output_length)

        if block_setting.nodes[-1] == -1:
            output_key = block_setting.output_key
            if output_key is None:
                if isinstance(candidate_output_node, dict):
                    raise ValueError(
                        'Output is dict. Plese specify output_key to the '
                        f"last nodes: {block_setting}")
                output_node = int(candidate_output_node)
            else:
                if block_setting.is_last:
                    output_length = output_length
                    output_node = int(output_length[output_key])
                else:
                    raise ValueError(
                        'Cannot put output_key when is_last is False: '
                        f"{block_setting}")
        else:
            output_node = block_setting.nodes[-1]

        if output_node == -1:
            raise ValueError(
                f"Output node inference failed for: {block_setting}")

        return input_node, output_node

    @classmethod
    def _get_n_input_node(
            cls, block_setting, predecessors, dict_block_setting,
            input_length):
        return dict_block_setting[tuple(predecessors)[0]].nodes[-1]

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length):
        if cls.is_trainable():
            return output_length
        else:
            return input_node

    def __init__(
            self, block_setting, *,
            create_linears=True, create_activations=True, create_dropouts=True,
            no_parameter=False, residual_dimension=None, **kwargs):
        super().__init__()
        self.block_setting = block_setting
        self.residual = self.block_setting.residual
        self.coeff = self.block_setting.coeff

        if no_parameter:
            create_linears = False
            create_activations = False
            create_dropouts = False
            self.activation = self.create_activation()

        if create_linears:
            self.linears = self.create_linears()

        if create_activations:
            self.activations = self.create_activations()

        if create_dropouts:
            self.dropout_ratios = self.create_dropout_ratios()

        if self.residual:
            nodes = block_setting.nodes
            if residual_dimension is None:
                residual_dimension = nodes[-1]
            if nodes[0] == residual_dimension:
                self.shortcut = acts.identity
            else:
                if self.block_setting.allow_linear_residual:
                    bias = self.block_setting.bias
                    self.shortcut = torch.nn.Linear(
                        nodes[0], residual_dimension, bias=bias)
                else:
                    raise ValueError(
                        'Residual input and output sizes differs for: '
                        f"{self.block_setting}")
        return

    def create_linears(self, nodes=None, bias=None):
        if nodes is None:
            nodes = self.block_setting.nodes
        if bias is None:
            bias = self.block_setting.bias

        try:
            linears = torch.nn.ModuleList([
                torch.nn.Linear(n1, n2, bias=bias)
                for n1, n2 in zip(nodes[:-1], nodes[1:])])
        except RuntimeError:
            raise ValueError(f"Cannot cretate linear for {self.block_setting}")
        return linears

    def create_activation(self, activation_settings=None):
        if activation_settings is None:
            activation_settings = self.block_setting.activations
        if len(activation_settings) != 1:
            raise ValueError(
                f"Invalid activation length: {len(activation_settings)} "
                f"for {self.block_setting}")
        return acts.DICT_ACTIVATIONS[activation_settings[0]]

    def create_activations(self, activation_settings=None, residual=None):
        if activation_settings is None:
            activation_settings = self.block_setting.activations
        if residual is None:
            residual = self.block_setting.residual

        list_activations = [
            acts.DICT_ACTIVATIONS[activation]
            for activation in self.block_setting.activations]
        if self.residual:
            activations = list_activations[:-1] \
                + [acts.DICT_ACTIVATIONS['identity']] \
                + [list_activations[-1]]
        else:
            activations = list_activations
        return activations

    def create_dropout_ratios(self, dropouts=None):
        if dropouts is None:
            dropouts = self.block_setting.dropouts
        dropout_ratios = [
            dropout_ratio for dropout_ratio in dropouts]
        return dropout_ratios

    def forward(self, x, supports=None, original_shapes=None):
        h = self._forward_core(
            x, supports=supports, original_shapes=original_shapes)
        if self.residual:
            if self.block_setting.activation_after_residual:
                return self.activations[-1](h + self.shortcut(x))
            else:
                return self.activations[-1](h) + self.shortcut(x)
        else:
            return h

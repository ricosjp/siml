
import numpy as np
import torch

from .. import setting
from . import network
from . import siml_module


class Group(siml_module.SimlModule):

    @staticmethod
    def get_name():
        return 'group'

    @staticmethod
    def is_trainable():
        return True

    @staticmethod
    def accepts_multiple_inputs():
        return True

    @staticmethod
    def uses_support():
        return True

    @classmethod
    def _get_n_input_node(
            cls, block_setting, predecessors, dict_block_setting,
            input_length, model_setting):
        return cls.sum_dim_if_needed(cls.create_group_setting(
            block_setting, model_setting).input_length)

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length, model_setting):
        return cls.sum_dim_if_needed(cls.create_group_setting(
            block_setting, model_setting).output_length)

    @staticmethod
    def create_group_setting(block_setting, model_setting):
        list_group_setting = [
            g for g in model_setting.groups if g.name == block_setting.name]
        if len(list_group_setting) != 1:
            raise ValueError(
                f"{len(list_group_setting)} group setting found. "
                'Use the name for the group.name and block.name')
        return list_group_setting[0]

    @staticmethod
    def sum_dim_if_needed(dim):
        if isinstance(dim, (int, np.int32, np.int64)):
            return dim
        elif isinstance(dim, dict):
            return np.sum([v for v in dim.values()])
        else:
            raise ValueError(
                f"Unexpected dimension format: {dim} ({dim.__class__})")

    def __init__(self, block_setting, model_setting):
        """Initialize the NN.

        Parameters
        -----------
        block_setting: siml.setting.BlockSetting
            BlockSetting object.
        model_setting: siml.setting.ModelSetting
            ModeliSetting object that is fed to the Network object.
        """
        super().__init__(block_setting, create_linears=False)
        self.group_setting = self.create_group_setting(
            block_setting, model_setting)
        self.group = self._create_group(block_setting, model_setting)
        return

    def _create_group(self, block_setting, model_setting):
        model_setting = setting.ModelSetting(blocks=self.group_setting.blocks)
        trainer_setting = setting.TrainerSetting(
            inputs=self.group_setting.inputs,
            outputs=self.group_setting.outputs,
            support_input=self.group_setting.support_inputs)
        return network.Network(model_setting, trainer_setting)

    def forward(self, x, supports, original_shapes=None):
        return self.generate_outputs(self.group({
            'x': x, 'supports': supports, 'original_shapes': original_shapes}))

    def generate_inputs(self, dict_hidden):
        if isinstance(self.input_names, (list, tuple)):
            return torch.cat([
                dict_hidden[k] for k in self.input_names], dim=-1)
        else:
            return {
                k: torch.cat([
                    dict_hidden[v_] for v_ in v], dim=-1)
                for k, v in self.input_names.items()}

    def generate_outputs(self, y):
        if isinstance(self.output_names, (list, tuple)):
            return y
        elif isinstance(self.output_names, dict):
            raise ValueError(y)

    @property
    def input_names(self):
        return self.group_setting.input_names

    @property
    def output_names(self):
        return self.group_setting.output_names


import torch

from . import siml_module

from siml.util import debug_if_necessary


class Reshape(siml_module.SimlModule):
    """Reshape block."""

    @staticmethod
    def get_name():
        return 'reshape'

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return False

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length, **kwargs):
        return block_setting.optional['new_shape'][-1]

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        self.new_shape = block_setting.optional['new_shape']
        return

    @debug_if_necessary
    def forward(self, x, supports=None, original_shapes=None, **kwargs):
        return torch.reshape(x, self.new_shape)


class FeaturesToTimeSeries(siml_module.SimlModule):
    """Convert feature axes to time series."""

    @staticmethod
    def get_name():
        return 'features_to_time_series'

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return False

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length, **kwargs):
        return 1

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        return

    @debug_if_necessary
    def forward(self, x, supports=None, original_shapes=None, **kwargs):
        axes = list(range(len(x.shape)))
        return torch.permute(x, [-1] + axes[:-1])[..., None]


class TimeSeriesToFeatures(siml_module.SimlModule):
    """Convert time series axes to features."""

    @staticmethod
    def get_name():
        return 'time_series_to_features'

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return False

    @classmethod
    def _get_n_input_node(
            cls, block_setting, predecessors, dict_block_setting,
            input_length, **kwargs):
        return 1

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length, **kwargs):
        return 1

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)

        # if not block_setting.is_last:
        #     raise ValueError(f"Should be at last: {block_setting}")
        return

    @debug_if_necessary
    def forward(self, x, supports=None, original_shapes=None, **kwargs):
        axes = list(range(len(x.shape)))
        return torch.reshape(
            torch.permute(x, axes[1:] + [0]), list(x.shape[1:-1]) + [-1])


class Accessor(siml_module.SimlModule):
    """Access data using the given index."""

    @staticmethod
    def get_name():
        return 'accessor'

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return False

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        self.index = block_setting.optional.get('index', 0)
        return

    @debug_if_necessary
    def forward(self, x, supports=None, original_shapes=None, **kwargs):
        return x[self.index]

from . import siml_module


class Identity(siml_module.SimlModule):
    """Identity block."""

    @staticmethod
    def get_name():
        return 'identity'

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
            input_length):
        return input_length

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length):
        return input_node

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        return

    def forward(self, *x, supports=None, original_shapes=None):
        if len(x) == 1:
            return x[0]
        else:
            return x

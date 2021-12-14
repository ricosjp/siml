
from . import siml_module


class Share(siml_module.SimlModule):
    """The same module as the specified reference."""

    @staticmethod
    def get_name():
        return 'share'

    @staticmethod
    def is_trainable():
        return True

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
        return dict_block_setting[block_setting.reference_block_name].nodes[0]

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length, **kwargs):
        return dict_block_setting[block_setting.reference_block_name].nodes[-1]

    def __init__(self, block_setting, reference_block):
        super().__init__(
            block_setting, create_linears=False,
            create_activations=False, create_dropouts=False,
            no_parameter=True)
        self.reference_block = reference_block

        return

    @property
    def linears(self):
        return self.reference_block.linears

    def _forward_core(self, x, supports=None, original_shapes=None):
        """Execute the NN's forward computation.

        Parameters
        ----------
        x: torch.Tensor
            Input of the NN.

        Returns
        -------
        y: torch.Tensor
            Output of the NN.
        """
        return self.reference_block(x)

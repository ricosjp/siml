
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

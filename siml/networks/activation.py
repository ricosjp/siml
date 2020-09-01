
from . import siml_module


class Activation(siml_module.SimlModule):
    """Activation block."""

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        self.use_original_shapes = self.block_setting.activations[0] in [
            'max_pool', 'max', 'mean']
        self.dict_key = block_setting.optional.get('dict_key', None)
        return

    def forward(self, x, supports=None, original_shapes=None):
        if self.use_original_shapes:
            if self.dict_key is None:
                return self.activation(x, original_shapes)
            else:
                return self.activation(x, original_shapes[self.dict_key])
        else:
            return self.activation(x)

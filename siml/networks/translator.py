
import torch

from . import siml_module
from . import activations


class Translator(siml_module.SimlModule):
    """Translation block."""

    @staticmethod
    def get_name():
        return 'translator'

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def uses_support():
        return False

    def __init__(self, block_setting):
        super().__init__(block_setting, no_parameter=True)
        self.method = block_setting.optional.get('method', 'mean')
        self.components = block_setting.optional.get('components', None)
        if self.method == 'mean':
            self.aggregate_func = torch.mean
        elif self.method == 'min':
            self.aggregate_func = activations.min_func
        else:
            raise ValueError(f"Unexpected 'method': {self.method}")

        if self.components is None:
            self.forward = self._forward_all_components
        else:
            self.forward = self._forward_partial_components

        return

    def _forward_all_components(self, x, supports=None, original_shapes=None):
        split_x = activations.split(x, original_shapes)
        return torch.cat([
            s - self.aggregate_func(s, dim=0, keepdims=True)
            for s in split_x], dim=0)

    def _forward_partial_components(
            self, x, supports=None, original_shapes=None):
        split_x = activations.split(x, original_shapes)
        return torch.cat([s - self._aggregate(s) for s in split_x], dim=0)

    def _aggregate(self, x):
        return torch.stack(
            [
                self.aggregate_func(x[..., i], dim=0, keepdims=False)
                if i in self.components
                else torch.zeros(x.shape[1:-1]).to(x.device)
                for i in range(x.shape[-1])],
            dim=-1)

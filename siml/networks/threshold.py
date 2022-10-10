from . import siml_module
import torch


class Threshold(siml_module.SimlModule):
    @staticmethod
    def get_name():
        return 'threshold'

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return False

    DEFALUT_THRESHOLD = 0.0

    def __init__(self, block_setting):
        super().__init__(block_setting,
                         create_activations=False,
                         no_parameter=True)
        self.threshold = self._select_threshold(block_setting)
        self.value = self._select_value(block_setting)

        self._func = torch.nn.modules.activation.Threshold(
            threshold=self.threshold,
            value=self.value,
        )

    def _select_threshold(self, block_setting) -> float:
        if 'threshold' not in block_setting.optional:
            return self.DEFALUT_THRESHOLD

        return block_setting.optional['threshold']

    def _select_value(self, block_setting) -> float:
        if 'value' not in block_setting.optional:
            return self._select_threshold(block_setting)

        return block_setting.optional['value']

    def forward(self,
                x: torch.Tensor,
                supports=None,
                original_shapes=None) -> torch.Tensor:
        out = self._func(x)
        return out

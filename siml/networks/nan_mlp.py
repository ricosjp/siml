
import torch

from . import mlp


class NaNMLP(mlp.MLP):
    """Multi Layer Perceptron with NaN handling."""

    @staticmethod
    def get_name():
        return 'nan_mlp'

    @staticmethod
    def is_trainable():
        return True

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return False

    def __init__(self, block_setting, **kwargs):
        super().__init__(block_setting, **kwargs)
        self.pad_value = self.block_setting.optional.get('pad_value', 0.)
        self.axis = self.block_setting.optional.get('axis', 0)
        if self.axis < 0 or self.axis > 1:
            raise ValueError(
                f"optional: axis should be 0 or 1 for {self.block_setting}")
        return

    def _forward_core(self, x, supports=None, original_shapes=None):
        """Execute the NN's forward computation.

        Parameters
        -----------
        x: numpy.ndarray or cupy.ndarray
            Input of the NN.

        Returns
        --------
        y: numpy.ndarray or cupy.ndarray
            Output of the NN.
        """
        x_shape = x.shape
        y_shape = list(x_shape[:-1]) + [self.block_setting.nodes[-1]]
        y = torch.ones(*y_shape, device=x.device) * self.pad_value

        if self.axis == 0:
            not_nan_filter = torch.einsum('i...->i', torch.isnan(x)) < .1
            h = x[not_nan_filter]
        elif self.axis == 1:
            not_nan_filter = torch.einsum('ti...->i', torch.isnan(x)) < .1
            h = x[:, not_nan_filter]
        else:
            raise ValueError('Should not rearch here')

        h = super()._forward_core(h)

        if self.axis == 0:
            y[not_nan_filter] = h
        elif self.axis == 1:
            y[:, not_nan_filter] = h
        else:
            raise ValueError('Should not rearch here')

        return y

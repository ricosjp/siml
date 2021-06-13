
import torch

from . import siml_module


class Dirichlet(siml_module.SimlModule):
    """Dirichlet boundary condition management."""

    def __init__(self, block_setting):
        """Initialize the module.

        Parameters
        -----------
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
        """
        super().__init__(
            block_setting, no_parameter=True, create_activations=False)
        return

    def forward(
            self, *xs, supports=None, original_shapes=None):
        """
        Take into account Dirichlet boundary condition.

        Parameters
        ----------
        xs: List[torch.Tensor]
            0: Variable values
            1: Dirichlet values.

        Returns
        -------
        ys: torch.Tensor
            Variable values with Dirichlet.
        """
        if len(xs) != 2:
            raise ValueError(
                f"Input should be x and Dirichlet ({len(xs)} given)")
        x = xs[0]
        dirichlet = xs[1]
        filter_not_nan = ~ torch.isnan(dirichlet)
        x[filter_not_nan] = dirichlet[filter_not_nan]
        return x


class NeumannIsoGCN(siml_module.SimlModule):
    """Neumann boundary condition management using IsoGCN."""

    def __init__(self, block_setting):
        """Initialize the module.

        Parameters
        -----------
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
        """
        super().__init__(
            block_setting, no_parameter=True, create_activations=False)
        return

    def forward(
            self, *xs, supports=None, original_shapes=None):
        """
        Take into account Neumann boundary condition using IsoGCN.

        Parameters
        ----------
        xs: List[torch.Tensor]
            0: Gradient values without Neumann.
            1: Neumann values multiplied with normal vectors.
            2: Inversed moment matrices.

        Returns
        -------
        ys: torch.Tensor
            Gradient values with Neumann.
        """
        if len(xs) != 3:
            raise ValueError(
                f"Input shoulbe x and Dirichlet ({len(xs)} given)")
        grad = xs[0]
        directed_neumann = xs[1]
        inversed_moment_tensors = xs[2]
        return grad + torch.einsum(
            'iklf,ilf->ikf',  inversed_moment_tensors, directed_neumann)

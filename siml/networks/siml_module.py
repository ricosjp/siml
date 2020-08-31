
import torch

from . import activations as acts


class SimlModule(torch.nn.Module):

    def __init__(
            self, block_setting, *,
            create_linears=True, create_activations=True, create_dropouts=True,
            no_parameter=False, residual_dimension=None, **kwargs):
        super().__init__()
        self.block_setting = block_setting
        self.residual = self.block_setting.residual
        self.coeff = self.block_setting.coeff

        if no_parameter:
            create_linears = False
            create_activations = False
            create_dropouts = False
            self.activation = self.create_activation()

        if create_linears:
            self.linears = self.create_linears()

        if create_activations:
            self.activations = self.create_activations()

        if create_dropouts:
            self.dropout_ratios = self.create_dropout_ratios()

        if self.residual:
            nodes = block_setting.nodes
            if residual_dimension is None:
                residual_dimension = nodes[-1]
            if nodes[0] == residual_dimension:
                self.shortcut = acts.identity
            else:
                if self.block_setting.allow_linear_residual:
                    bias = self.block_setting.bias
                    self.shortcut = torch.nn.Linear(
                        nodes[0], residual_dimension, bias=bias)
                else:
                    raise ValueError(
                        'Residual input and output sizes differs for: '
                        f"{self.block_setting}")
        return

    def create_linears(self, nodes=None, bias=None):
        if nodes is None:
            nodes = self.block_setting.nodes
        if bias is None:
            bias = self.block_setting.bias

        try:
            linears = torch.nn.ModuleList([
                torch.nn.Linear(n1, n2, bias=bias)
                for n1, n2 in zip(nodes[:-1], nodes[1:])])
        except RuntimeError:
            raise ValueError(f"Cannot cretate linear for {self.block_setting}")
        return linears

    def create_activation(self, activation_settings=None):
        if activation_settings is None:
            activation_settings = self.block_setting.activations
        if len(activation_settings) != 1:
            raise ValueError(
                f"Invalid activation length: {len(activation_settings)} "
                f"for {self.block_setting}")
        return acts.DICT_ACTIVATIONS[activation_settings[0]]

    def create_activations(self, activation_settings=None, residual=None):
        if activation_settings is None:
            activation_settings = self.block_setting.activations
        if residual is None:
            residual = self.block_setting.residual

        list_activations = [
            acts.DICT_ACTIVATIONS[activation]
            for activation in self.block_setting.activations]
        if self.residual:
            activations = list_activations[:-1] \
                + [acts.DICT_ACTIVATIONS['identity']] \
                + [list_activations[-1]]
        else:
            activations = list_activations
        return activations

    def create_dropout_ratios(self, dropouts=None):
        if dropouts is None:
            dropouts = self.block_setting.dropouts
        dropout_ratios = [
            dropout_ratio for dropout_ratio in dropouts]
        return dropout_ratios

    def forward(self, x, supports=None, original_shapes=None):
        h = self._forward_core(
            x, supports=supports, original_shapes=original_shapes)
        if self.residual:
            if self.block_setting.activation_after_residual:
                return self.activations[-1](h + self.shortcut(x))
            else:
                return self.activations[-1](h) + self.shortcut(x)
        else:
            return h

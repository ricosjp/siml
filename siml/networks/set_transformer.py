"""Set Transformer implementation based on https://arxiv.org/abs/1810.00825
and the official implementation https://github.com/juho-lee/set_transformer .
"""

import torch

from . import activations
from . import siml_module


class SetTransformerEncoder(siml_module.SimlModule):
    """Set Transformer's encoder based on
    https://arxiv.org/abs/1810.00825 .
    """

    @staticmethod
    def get_name():
        return 'set_transformer_encoder'

    @staticmethod
    def is_trainable():
        return True

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return False

    def __init__(self, block_setting):
        """Initialize the NN.

        Parameters
        -----------
        block_setting: siml.setting.BlockSetting
            BlockSetting object.
        """
        super().__init__(block_setting, create_linears=False)

        n_head = self.block_setting.optional.get(
            'n_head', 4)
        n_inducing_point = self.block_setting.optional.get(
            'n_inducing_point', 32)
        layer_norm = self.block_setting.optional.get('layer_norm', True)
        if n_inducing_point is None or n_inducing_point < 1:
            attention_block = SAB
        else:
            attention_block = ISAB

        nodes = self.block_setting.nodes
        activations = self.create_activations()
        self.blocks = torch.nn.ModuleList([
            attention_block(
                dim_in=n1, dim_out=n2, num_heads=n_head,
                n_inducing_point=n_inducing_point, layer_norm=layer_norm,
                activation=activation)
            for n1, n2, activation in zip(nodes[:-1], nodes[1:], activations)])
        self.dict_key = block_setting.optional.get('dict_key', None)
        return

    def _forward_core(self, x, supports=None, original_shapes=None):
        if self.dict_key is None:
            split_xs = activations.split(x, original_shapes)
        else:
            split_xs = activations.split(x, original_shapes[self.dict_key])
        h = torch.cat([
            self._forward_one_sample(split_x)
            for split_x in split_xs], dim=0)
        return h

    def _forward_one_sample(self, x):
        h = x[None, ...]
        for block in self.blocks:
            h = block(h)
        return h[0]


class SetTransformerDecoder(siml_module.SimlModule):
    """Set Transformer's encoder based on
    https://arxiv.org/abs/1810.00825 .
    """

    @staticmethod
    def get_name():
        return 'set_transformer_decoder'

    @staticmethod
    def is_trainable():
        return True

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return False

    def __init__(self, block_setting):
        """Initialize the NN.

        Parameters
        -----------
        block_setting: siml.setting.BlockSetting
            BlockSetting object.
        """
        super().__init__(block_setting, create_linears=False)

        n_head = self.block_setting.optional.get(
            'n_head', 4)
        layer_norm = self.block_setting.optional.get('layer_norm', True)
        self.n_input = self.block_setting.optional.get('n_input', None)
        n_output = self.block_setting.optional.get('n_output', 1)
        self.dict_key = block_setting.optional.get('dict_key', None)

        if len(self.block_setting.nodes) != 2:
            raise ValueError(
                f"len(nodes) should be 2 for: {self.block_setting}")
        n1 = self.block_setting.nodes[0]
        n2 = self.block_setting.nodes[1]
        activation = self.create_activation()
        self.decoder = torch.nn.Sequential(
            PMA(
                n1, n_head, n_output, layer_norm=layer_norm,
                activation=activation),
            SAB(n1, n1, n_head, layer_norm=layer_norm, activation=activation),
            SAB(n1, n1, n_head, layer_norm=layer_norm, activation=activation),
            torch.nn.Linear(n1, n2))

        return

    def _forward_core(self, x, supports=None, original_shapes=None):
        if self.n_input is None:
            if self.dict_key is None:
                split_xs = activations.split(x, original_shapes)
            else:
                split_xs = activations.split(x, original_shapes[self.dict_key])
            h = torch.cat([
                self.decoder(split_x[None, ...])[0]
                for split_x in split_xs], dim=0)
        else:
            split_xs = torch.split(x, self.n_input)
            h = torch.cat([
                self.decoder(split_x[None, ...])[0]
                for split_x in split_xs], dim=0)
        return h


class MAB(torch.nn.Module):
    """MAB: the Multihead Attention Block.
    It maps [n, d_in] x [n, d_in] -> [n, d_out].
    """

    def __init__(
            self, dim_Q, dim_K, dim_V, num_heads, activation, layer_norm):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = torch.nn.Linear(dim_Q, dim_V)
        self.fc_k = torch.nn.Linear(dim_K, dim_V)
        self.fc_v = torch.nn.Linear(dim_K, dim_V)
        self.layer_norm = layer_norm
        if activation is None:
            self.activation = activations.identity
        else:
            self.activation = activation

        if self.layer_norm:
            self.layer_norm0 = torch.nn.LayerNorm(dim_V)
            self.layer_norm1 = torch.nn.LayerNorm(dim_V)
        self.fc_o = torch.nn.Linear(dim_V, dim_V)
        return

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / self.dim_V**.5, 2)
        H = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        if self.layer_norm:
            H = self.layer_norm0(H)
        H = H + self.activation(self.fc_o(H))
        if self.layer_norm:
            H = self.layer_norm1(H)
        return H


class SAB(torch.nn.Module):
    """SAB: the Set Attention Block.
    It maps [n, d_in] -> [n, d_out], without inducing points,
    therefore it takes O(n^2).
    """

    def __init__(
            self, dim_in, dim_out, num_heads,
            layer_norm, activation, n_inducing_point=None):
        super(SAB, self).__init__()
        self.mab = MAB(
            dim_in, dim_in, dim_out, num_heads, layer_norm=layer_norm,
            activation=activation)
        return

    def forward(self, X):
        return self.mab(X, X)


class ISAB(torch.nn.Module):
    """ISAB: the Induced Set Attention Block.
    It maps [n, d_in] -> [n, d_out], with inducing points,
    therefore it takes O(n n_induce), where n_induce is the number of
    the inducing points.
    """

    def __init__(
            self, dim_in, dim_out, num_heads,
            layer_norm, activation, n_inducing_point=None):
        super(ISAB, self).__init__()
        self.I_ = torch.nn.Parameter(
            torch.Tensor(1, n_inducing_point, dim_out))
        torch.nn.init.xavier_uniform_(self.I_)
        self.mab0 = MAB(
            dim_out, dim_in, dim_out, num_heads, layer_norm=layer_norm,
            activation=activation)
        self.mab1 = MAB(
            dim_in, dim_out, dim_out, num_heads, layer_norm=layer_norm,
            activation=activation)
        return

    def forward(self, X):
        H = self.mab0(self.I_.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(torch.nn.Module):
    """PMA: Pooling by Multihead Attention.
    It maps [n, d_in] -> [k, d_out], where k is the number of the output
    vectors.
    """

    def __init__(
            self, dim, num_heads, num_seeds, activation, layer_norm):
        super(PMA, self).__init__()
        self.S = torch.nn.Parameter(torch.Tensor(1, num_seeds, dim))
        torch.nn.init.xavier_uniform_(self.S)
        self.mab = MAB(
            dim, dim, dim, num_heads, layer_norm=layer_norm,
            activation=activation)
        return

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

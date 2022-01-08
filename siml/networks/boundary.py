
import torch

from . import activations
from . import siml_module


class Dirichlet(siml_module.SimlModule):
    """Dirichlet boundary condition management."""

    @staticmethod
    def get_name():
        return 'dirichlet'

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def accepts_multiple_inputs():
        return True

    @staticmethod
    def uses_support():
        return False

    @classmethod
    def _get_n_input_node(
            cls, block_setting, predecessors, dict_block_setting,
            input_length, **kwargs):
        return dict_block_setting[predecessors[0]].nodes[-1]

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length, **kwargs):
        return input_node

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
        x[..., filter_not_nan] = dirichlet[filter_not_nan]
        return x


class NeumannIsoGCN(siml_module.SimlModule):
    """Neumann boundary condition management using IsoGCN."""

    @staticmethod
    def get_name():
        return 'neumann_isogcn'

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def accepts_multiple_inputs():
        return True

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
        """Initialize the module.

        Parameters
        -----------
            block_setting: siml.setting.BlockSetting
                BlockSetting object.
        """
        super().__init__(
            block_setting, no_parameter=True, create_activations=False)
        self.reference_block = reference_block
        self.create_neumann_linear = self.block_setting.optional.get(
            'create_neumann_linear', False)
        self.neumann_factor = self.block_setting.optional.get(
            'neumann_factor', 1.)
        self.create_neumann_ratio = self.block_setting.optional.get(
            'create_neumann_ratio', False)
        if self.reference_block is None:
            raise ValueError(f"Feed reference_block for: {block_setting}")
        if self.reference_block.create_subchain:
            if len(self.reference_block.subchains) != 1 \
                    or len(self.reference_block.subchains[0]) != 1:
                raise ValueError(
                    'Subchain setting incorrect for '
                    f"{self.reference_block.block_setting} "
                    f"(referenced from: {self.block_setting})")

            if self.create_neumann_linear:
                self.linear = torch.nn.Linear(
                    *self.reference_block.subchains[0][0].weight.shape,
                    bias=False)
            else:
                self.linear = self.reference_block.subchains[0][0]

            if self.linear.bias is not None:
                raise ValueError(
                    'Reference IsoGCN should have no bias: '
                    f"{self.reference_block.block_setting}")
        else:
            self.linear = activations.identity

        if self.create_neumann_ratio:
            self.coeff = torch.nn.Linear(1, 1, bias=False)
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
        neumann = torch.einsum(
            'ikl,il...f->ik...f',
            inversed_moment_tensors[..., 0],
            self.linear(directed_neumann)) * self.neumann_factor
        if self.create_neumann_ratio:
            sigmoid_coeff = torch.sigmoid(self.coeff.weight[0, 0])
            return (sigmoid_coeff * grad + (1 - sigmoid_coeff) * neumann) * 2
        else:
            return grad + neumann


class NeumannEncoder(siml_module.SimlModule):
    """Encoder for Neumann condition."""

    @staticmethod
    def get_name():
        return 'neumann_encoder'

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def accepts_multiple_inputs():
        return True

    @staticmethod
    def uses_support():
        return False

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length, **kwargs):
        return dict_block_setting[block_setting.reference_block_name].nodes[-1]

    def __init__(self, block_setting, reference_block):
        super().__init__(
            block_setting, create_neumann_linears=False,
            create_activations=False, create_dropouts=False,
            no_parameter=True)
        self.epsilon = self.block_setting.optional.get('epsilon', 1.e-5)
        self.reference_block = reference_block
        if self.reference_block is None:
            raise ValueError(f"Feed reference_block for: {block_setting}")
        self.activation_names = [
            name for name in self.reference_block.block_setting.activations]
        self.derivative_activations = [
            self._define_activation_derivative(name)
            for name in self.activation_names]
        return

    @property
    def linears(self):
        return self.reference_block.linears

    @property
    def weights(self):
        return [linear.weight for linear in self.linears]

    @property
    def activations(self):
        return self.reference_block.activations

    def forward(
            self, *xs, supports=None, original_shapes=None):
        """
        Take into account Neumann boundary condition using IsoGCN.

        Parameters
        ----------
        xs: List[torch.Tensor]
            0: Variable values (not encoded)
            1: Neumann values multiplied with normal vectors
            or
            0: Variable values (not encoded)
            1: Neumann values
            2: Surface weighted normal vectors

        Returns
        -------
        ys: torch.Tensor
            Embedded Neumann values multiplied with normal vectors.
        """
        if len(xs) == 2:  # Neumann * normal
            h = xs[0]
            neumann = xs[1]
            if len(self.linears) > 1 \
                    or self.activations[0] != activations.identity \
                    or self.linears[0].bias is not None:
                raise ValueError(
                    f"Nonlinear MLP found for: {self.block_setting}")
        elif len(xs) == 3:  # Neumann value without normal vector
            h = xs[0]
            neumann = xs[1]
            surface_normals = xs[2]
        else:
            raise ValueError(
                f"Input shoulbe x and Neumann (and normal) ({len(xs)} given)")

        with torch.no_grad():
            for linear, name, activation, derivative_activation in zip(
                    self.linears, self.activation_names,
                    self.activations, self.derivative_activations):
                if name == 'identity':
                    neumann = torch.einsum(
                        'i...f,fg->i...g', neumann, linear.weight.T)
                else:
                    lineared_h = linear(h)
                    derivative_h = derivative_activation(lineared_h)
                    neumann = torch.einsum(
                        'ig,i...g->i...g', derivative_h, torch.einsum(
                            'i...f,fg->i...g', neumann, linear.weight.T))
                    h = activation(lineared_h)

        if len(xs) == 2:
            return neumann
        else:
            return torch.einsum(
                'i...f,ik->ik...f', neumann, surface_normals[..., 0])

    def _define_activation_derivative(self, name):
        if name == 'identity':
            return activations.one
        elif name == 'tanh':
            return activations.derivative_tanh
        elif name == 'leaky_relu':
            return activations.DerivativeLeakyReLU()
        else:
            raise ValueError(f"Unsupported activation name: {name}")


class Interaction(siml_module.SimlModule):
    """Interaction layer."""

    @staticmethod
    def get_name():
        return 'interaction'

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def accepts_multiple_inputs():
        return True

    @staticmethod
    def uses_support():
        return True

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length, **kwargs):
        return input_node

    def __init__(self, block_setting):
        super().__init__(
            block_setting,
            create_activations=False, create_dropouts=False,
            no_parameter=True)
        if 'operator' in block_setting.optional:
            str_op = block_setting.optional['operator']
            if str_op == 'add':
                self.op = torch.add
            elif str_op == 'mul':
                self.op = torch.mul
            elif str_op == 'sub':
                self.op = torch.sub
            else:
                raise ValueError(f"Unknown operator for reducer: {str_op}")
        else:
            self.op = torch.add
            self.block_setting.optional['operator'] = 'add'
            print(f"optional.operator = add is set for: {block_setting}")

        self.transpose = self.block_setting.optional.get('transpose', False)
        self.other_zero = self.block_setting.optional.get('other_zero', True)
        self.factor = self.block_setting.optional.get('factor', 1.)
        return

    def forward(
            self, *xs, supports=None, original_shapes=None):
        """
        Compute interactions between two meshes.

        Parameters
        ----------
        xs: List[torch.Tensor]
            0: self values (length = n)
            or
            0: self values (length = n)
            1: other values (lengh = m)
        supports: List
            0: incidence matrix from other to self (n x m matrix)

        Returns
        -------
        ys: torch.Tensor
            Interaction value (length = n)
        """
        if len(xs) == 1:
            x = xs[0]
            other = xs[0]
        elif len(xs) == 2:
            x = xs[0]
            other = xs[1]
        else:
            raise ValueError(f"Feed one or two variables ({len(xs)} is fed)")

        if self.other_zero:
            ret = torch.zeros(x.shape, device=x.device)
        else:
            ret = x

        if self.transpose:
            incidence = supports[0].transpose(0, 1)
        else:
            incidence = supports[0]
        row = incidence._indices()[0]
        ret[row] = self.op(incidence.mm(other)[row], x[row]) * self.factor
        return ret


class Assignment(siml_module.SimlModule):
    """Assignment layer."""

    @staticmethod
    def get_name():
        return 'assignment'

    @staticmethod
    def is_trainable():
        return False

    @staticmethod
    def accepts_multiple_inputs():
        return True

    @staticmethod
    def uses_support():
        return True

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length, **kwargs):
        return input_node

    def __init__(self, block_setting):
        super().__init__(
            block_setting,
            create_activations=False, create_dropouts=False,
            no_parameter=True)
        self.broadcast = self.block_setting.optional.get('broadcast', False)
        self.dict_key = self.block_setting.optional.get('dict_key', None)
        return

    def forward(
            self, *xs, supports=None, original_shapes=None):
        """
        Assign values.

        Parameters
        ----------
        xs: List[torch.Tensor]
            0: destination values
            1: source values
            2: condition values

        Returns
        -------
        ys: torch.Tensor
            Values assigned
        """
        if len(xs) == 3:
            x = xs[0]
            other = xs[1]
            cond_val = xs[2]
        else:
            raise ValueError(f"Feed three variables ({len(xs)} is fed)")

        if cond_val.shape[-1] == 1:
            cond = cond_val[..., 0] > .5
        else:
            cond = cond_val > .5

        if self.broadcast:
            if self.dict_key is None:
                split_data = activations.split(x, original_shapes)
                split_cond = activations.split(cond, original_shapes)
            else:
                split_data = activations.split(
                    x, original_shapes[self.dict_key])
                split_cond = activations.split(
                    cond, original_shapes[self.dict_key])
            y = torch.cat([
                split_data[i_other] * ~split_cond[i_other][..., None]
                + other[[i_other]] * split_cond[i_other][..., None]
                for i_other in range(len(other))], dim=0)

        else:
            x[cond] = other[cond]
            y = x

        return y

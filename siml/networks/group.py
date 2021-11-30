
import numpy as np
import torch

from .. import setting
from .. import util
from . import network
from . import siml_module


class Group(siml_module.SimlModule):

    @staticmethod
    def get_name():
        return 'group'

    @staticmethod
    def is_trainable():
        return True

    @staticmethod
    def accepts_multiple_inputs():
        return True

    @staticmethod
    def uses_support():
        return True

    @classmethod
    def _get_n_input_node(
            cls, block_setting, predecessors, dict_block_setting,
            input_length, model_setting):
        return cls.sum_dim_if_needed(cls.create_group_setting(
            block_setting, model_setting).input_length)

    @classmethod
    def _get_n_output_node(
            cls, input_node, block_setting, predecessors, dict_block_setting,
            output_length, model_setting):
        return cls.sum_dim_if_needed(cls.create_group_setting(
            block_setting, model_setting).output_length)

    @staticmethod
    def create_group_setting(block_setting, model_setting):
        list_group_setting = [
            g for g in model_setting.groups if g.name == block_setting.name]
        if len(list_group_setting) != 1:
            raise ValueError(
                f"{len(list_group_setting)} group setting found. "
                'Use the name for the group.name and block.name')
        return list_group_setting[0]

    @staticmethod
    def sum_dim_if_needed(dim):
        if isinstance(dim, (int, np.int32, np.int64)):
            return dim
        elif isinstance(dim, dict):
            return np.sum([v for v in dim.values()])
        else:
            raise ValueError(
                f"Unexpected dimension format: {dim} ({dim.__class__})")

    def __init__(self, block_setting, model_setting):
        """Initialize the NN.

        Parameters
        -----------
        block_setting: siml.setting.BlockSetting
            BlockSetting object.
        model_setting: siml.setting.ModelSetting
            ModeliSetting object that is fed to the Network object.
        """
        super().__init__(block_setting, create_linears=False)
        self.group_setting = self.create_group_setting(
            block_setting, model_setting)
        self.group = self._create_group(block_setting, model_setting)
        self.loop = self.group_setting.repeat > 1
        self.mode = self.group_setting.mode
        self.debug = self.group_setting.debug
        self.componentwise_alpha = self.group_setting.optional.get(
            'componentwise_alpha', False)

        if self.loop:
            input_is_dict = isinstance(
                self.group_setting.inputs.variables, dict)
            output_is_dict = isinstance(
                self.group_setting.outputs.variables, dict)
            if (input_is_dict and not output_is_dict) \
                    or (not input_is_dict and output_is_dict):
                raise ValueError(
                    'When loop, both inputs and outputs should be '
                    'either list or dict.\n'
                    f"inputs:\n{self.group_setting.inputs}\n"
                    f"outputs:\n{self.group_setting.outputs}")
            self.skips = self.group_setting.inputs.collect_values(
                'skip', default=False)
            self.mask_function = util.VariableMask(
                skips=self.skips,
                dims=self.group_setting.inputs.dims,
                is_dict=output_is_dict)
            if self.mode == 'implicit':
                if self.group_setting.convergence_threshold is None:
                    raise ValueError(
                        'Feed convergence_threshold to the GroupSetting '
                        'when mode == "implicit"')
                self.forward = self.forward_implicit
            elif self.mode == 'steady':
                raise NotImplementedError
            elif self.mode == 'simple':
                self.forward = self.forward_w_loop
            else:
                raise ValueError(f"Unexpected mode: {self.mode}")
        else:
            self.forward = self.forward_wo_loop

        if 'residual' in self.block_setting.losses:
            self.residual_loss = True
            self.losses = {'residual': []}
        else:
            self.residual_loss = False

        return

    def _create_group(self, block_setting, model_setting):
        model_setting = setting.ModelSetting(blocks=self.group_setting.blocks)
        trainer_setting = setting.TrainerSetting(
            inputs=self.group_setting.inputs,
            outputs=self.group_setting.outputs,
            support_input=self.group_setting.support_inputs)
        return network.Network(model_setting, trainer_setting)

    def forward_wo_loop(self, x, supports, original_shapes=None):
        return self.group({
            'x': x, 'supports': supports,
            'original_shapes': original_shapes})

    def forward_w_loop(self, x, supports, original_shapes=None):
        h = x
        for i_repeat in range(self.group_setting.repeat):
            h_previous = self.mask_function(h)[0]
            h.update(self.group({
                'x': h, 'supports': supports,
                'original_shapes': original_shapes}))
            if self.group_setting.convergence_threshold is not None:
                residual = self.calculate_residual(
                    self.mask_function(h)[0], h_previous)
                if self.debug:
                    print(f"{i_repeat} {residual}")
                if residual < self.group_setting.convergence_threshold:
                    if self.debug:
                        print(f"Convergent ({i_repeat}: {residual})")
                    break

                if residual > 10:
                    if self.debug:
                        print(f"Divergent ({i_repeat}: {residual})")
                    break

        else:
            if self.group_setting.convergence_threshold is not None:
                print(
                    f"Not converged at in {self.group_setting.name} "
                    f"(residual = {residual})")
                pass
        return h

    def forward_implicit(self, x, supports, original_shapes=None):
        masked_x = self.mask_function(x, keep_empty_data=False)[0]
        h = x
        masked_h = self.mask_function(h, keep_empty_data=False)[0]
        masked_h_previous = masked_h
        operator = self.group({
            'x': h, 'supports': supports,
            'original_shapes': original_shapes})
        masked_operator = self.mask_function(
            operator, keep_empty_data=False)[0]
        masked_nabla_f = self.operate(self.operate(
            masked_h, masked_x, torch.sub), masked_operator, torch.sub)

        masked_nabla_f_previous = masked_nabla_f

        if self.debug:
            print('\n--\ni_repeat\tresidual\talpha')
        for i_repeat in range(self.group_setting.repeat):
            operator = self.group({
                'x': h, 'supports': supports,
                'original_shapes': original_shapes})
            masked_operator = self.mask_function(
                operator, keep_empty_data=False)[0]

            # R(u) = u - u(t) - D u dt
            masked_nabla_f = self.operate(self.operate(
                masked_h, masked_x, torch.sub), masked_operator, torch.sub)

            alphas = self._calculate_alphas(
                masked_h, masked_h_previous,
                masked_nabla_f, masked_nabla_f_previous)

            # - du = alpha_i R(u_i)
            masked_negative_dh = self.operate(
                alphas, masked_nabla_f, torch.mul)

            masked_h_previous = masked_h
            masked_nabla_f_previous = masked_nabla_f

            # h_{i+1} = h_i - (- du)
            masked_h = self.operate(masked_h, masked_negative_dh, torch.sub)

            # TODO: Validate with more variables
            h.update({k: masked_h[i] for i, k in enumerate(
                [k for k, v in self.skips.items() if ~np.all(v)])})

            residual = self.calculate_residual(
                masked_h, masked_h_previous)

            if self.debug:
                print(f"{i_repeat}\t{residual}\t{alphas}")
            if residual < self.group_setting.convergence_threshold:
                if self.debug:
                    print(f"Convergent ({i_repeat}: {residual})")
                break

            if residual > 100:
                print(f"Divergent ({i_repeat}: {residual})")
                if self.debug:
                    raise ValueError(f"Divergent ({i_repeat}: {residual})")
                break

        else:
            print(
                f"Not converged at in {self.group_setting.name} "
                f"(residual = {residual})")
            if self.debug:
                raise ValueError(
                    f"Not converged at in {self.group_setting.name} "
                    f"(residual = {residual})")
            pass

        if self.residual_loss:
            self.losses['residual'].append(residual)
        return h

    def _calculate_nabla_f(
            self, h, masked_h, masked_x, supports, original_shapes):
        operator = self.group({
            'x': h, 'supports': supports,
            'original_shapes': original_shapes})
        masked_operator = self.mask_function(
            operator, keep_empty_data=False)[0]

        # Ignore higher order derivatives
        nabla_f = self.operate(self.operate(
            masked_h, masked_x, torch.sub), masked_operator, torch.sub)
        return nabla_f

    def _calculate_alphas(self, h, h_previous, nabla_f, nabla_f_previous):
        """Compute the coefficient in the steepest gradient method based on
        Barzilai-Borwein method.
        """
        if isinstance(h, list):
            return [
                self._calculate_alphas(
                    h_, h_previous_, nabla_f_, nabla_f_previous_)
                for h_, h_previous_, nabla_f_, nabla_f_previous_
                in zip(h, h_previous, nabla_f, nabla_f_previous)]

        delta_h = h - h_previous
        delta_nabla_f = nabla_f - nabla_f_previous
        if torch.allclose(delta_h, torch.tensor(0.).to(delta_h.device)) \
                and torch.allclose(
                    delta_nabla_f, torch.tensor(0.).to(
                        delta_nabla_f.device)):
            alpha = 1.
        else:
            if self.componentwise_alpha:
                alpha = torch.abs(torch.einsum(
                    'n...,n...->...', delta_h, delta_nabla_f)) / (
                        torch.einsum(
                            'n...,n...->...', delta_nabla_f, delta_nabla_f)
                        + 1.e-5)
            else:
                alpha = torch.abs(torch.einsum(
                    'n...f,n...f->...', delta_h, delta_nabla_f)) / (
                        torch.einsum(
                            'n...f,n...f->...', delta_nabla_f, delta_nabla_f)
                        + 1.e-5)
        return alpha

    def operate(self, x, y, operator):
        if isinstance(x, list):
            assert len(x) == len(y), \
                f"Length not the same: {len(x)} vs {len(y)}"
            return [self.operate(x_, y_, operator) for x_, y_ in zip(x, y)]
        else:
            return operator(x, y)

    def calculate_residual(self, x, ref):
        if isinstance(x, list):
            assert len(x) == len(ref)
            return torch.sum(torch.stack([
                self.calculate_residual(x_, ref_)
                for x_, ref_ in zip(x, ref)]))
        else:
            return torch.linalg.norm(x - ref) \
                / (torch.linalg.norm(ref) + 1.e-5)

    def generate_inputs(self, dict_predecessors):
        if isinstance(self.input_names, (list, tuple)):
            return torch.cat([
                dict_predecessors[k] for k in self.input_names], dim=-1)
        else:
            return {
                k: torch.cat([
                    dict_predecessors[v_][k] for v_ in v
                    if k in dict_predecessors[v_]], dim=-1)
                for k, v in self.input_names.items()}

    def generate_outputs(self, y):
        if isinstance(self.output_names, (list, tuple)):
            return y
        elif isinstance(self.output_names, dict):
            raise ValueError(y)

    @property
    def input_names(self):
        return self.group_setting.input_names

    @property
    def output_names(self):
        return self.group_setting.output_names


import numpy as np
import torch

from .. import setting
from .. import util
from . import activations
from . import mlp
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
        self.time_series_length = self.group_setting.time_series_length
        self.time_series_mask = util.VariableMask(
            self.group_setting.inputs.time_series,
            self.group_setting.inputs.dims,
            invert=True)
        if isinstance(self.group_setting.inputs.time_series, list):
            self.time_series_keys = []
        elif isinstance(self.group_setting.inputs.time_series, dict):
            self.time_series_keys = [
                k for k, v in self.group_setting.inputs.time_series.items()
                if np.any(v)]
        else:
            raise ValueError(f"Invalid format: {self.group_setting.inputs}")
        self.debug = self.group_setting.debug

        self.componentwise_alpha = self.group_setting.optional.get(
            'componentwise_alpha', False)
        self.alpha_denominator_norm = self.group_setting.optional.get(
            'alpha_denominator_norm', True)
        self.divergent_threshold = self.group_setting.optional.get(
            'divergent_threshold', 100)
        self.learn_alpha = self.group_setting.optional.get(
            'learn_alpha', False)
        self.abs_alpha = self.group_setting.optional.get(
            'abs_alpha', True)
        self.steady = self.group_setting.optional.get(
            'steady', False)

        if self.abs_alpha:
            self.alpha_filter = torch.abs
        else:
            self.alpha_filter = activations.identity

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

            if self.learn_alpha:
                self.alpha_model = self._init_alpha_model()

            if self.mode == 'implicit':
                if self.group_setting.convergence_threshold is None:
                    raise ValueError(
                        'Feed convergence_threshold to the GroupSetting '
                        'when mode == "implicit"')
                forward = self.forward_implicit
            elif self.mode == 'steady':
                raise NotImplementedError
            elif self.mode == 'simple':
                forward = self.forward_w_loop
            else:
                raise ValueError(f"Unexpected mode: {self.mode}")
        else:
            forward = self.forward_wo_loop

        if self.time_series_length is None:
            self.forward = forward
        else:
            self.forward = self.forward_time_series
            self.forward_step = forward

        if 'residual' in self.block_setting.loss_names:
            self.residual_loss = True
            self.losses = {'residual': 0}
        else:
            self.residual_loss = False

        return

    def _init_alpha_model(self):
        if self.componentwise_alpha:
            alpha_model = {
                k:
                mlp.MLP(setting.BlockSetting(
                    nodes=[np.sum(v) * 3, np.sum(v)], activations=['sigmoid']))
                for k, v in self.mask_function.mask.items() if np.sum(v) > 0}
        else:
            alpha_model = {
                k:
                mlp.MLP(setting.BlockSetting(
                    nodes=[np.sum(v) * 3, 1], activations=['sigmoid']))
                for k, v in self.mask_function.mask.items() if np.sum(v) > 0}
        return torch.nn.ModuleDict(alpha_model)

    def _create_group(self, block_setting, model_setting):
        group_model_setting = setting.ModelSetting(
            blocks=self.group_setting.blocks, groups=model_setting.groups)
        trainer_setting = setting.TrainerSetting(
            inputs=self.group_setting.inputs,
            outputs=self.group_setting.outputs,
            support_input=self.group_setting.support_inputs)
        return network.Network(group_model_setting, trainer_setting)

    def forward_time_series(self, x, supports, original_shapes=None):
        ts = [x]
        if self.time_series_length < 0:
            time_series_length = self._get_time_series_length(x)
        else:
            time_series_length = self.time_series_length

        for i_time in range(time_series_length):
            ts.append(self.forward_step(
                self._generate_time_series_input(ts[-1], x, i_time),
                supports, original_shapes=original_shapes))
        return self._stack(ts[1:])

    def _get_time_series_length(self, x):
        if isinstance(x, dict):
            masked_x = self.time_series_mask(x, keep_empty_data=False)[0]
            lengths = np.array([len(v) for v in masked_x])
            if not np.all(lengths == lengths[0]):
                raise ValueError(
                    'Time length mismatch: '
                    f"{self.time_series_keys}, {lengths}")
            return lengths[0]
        else:
            raise NotImplementedError

    def _generate_time_series_input(self, x, original_x, time_index):
        if isinstance(x, dict):
            y = {
                k: original_x[k][time_index].clone()
                if k in self.time_series_keys
                else v.clone()
                for k, v in x.items()}
            # for v in y.values():
            #     if v.requires_grad:
            #         v.retain_grad()
            return y
        else:
            raise NotImplementedError

    def _stack(self, xs):
        if isinstance(xs[0], dict):
            return {
                k: torch.stack([x[k] for x in xs], dim=0)
                for k in xs[0].keys()}
        else:
            raise NotImplementedError

    def forward_wo_loop(self, x, supports, original_shapes=None):
        h = self.group({
            'x': x, 'supports': supports,
            'original_shapes': original_shapes})
        return h

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
            if self.group_setting.convergence_threshold is not None \
                    and self.debug:
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

            if self.steady:
                # R(u) = - D[u] dt
                masked_nabla_f = self.broadcast(-1, masked_operator, torch.mul)
            else:
                # R(u) = u - u(t) - D[u] dt
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

            if residual > self.divergent_threshold:
                print(f"Divergent ({i_repeat}: {residual})")
                if self.debug:
                    raise ValueError(f"Divergent ({i_repeat}: {residual})")
                break

        else:
            if self.debug:
                print(
                    f"Not converged at in {self.group_setting.name} "
                    f"(residual = {residual})")
            pass

        if self.residual_loss:
            self.losses['residual'] = residual

        # Add output values not involved in the loop
        h.update({
            k: operator[k] for k in operator.keys()
            if k not in self.skips})

        return h

    def _calculate_nabla_f(
            self, h, masked_h, masked_x, supports, original_shapes):
        operator = self.group({
            'x': h, 'supports': supports,
            'original_shapes': original_shapes})
        masked_operator = self.mask_function(
            operator, keep_empty_data=False)[0]

        nabla_f = self.operate(self.operate(
            masked_h, masked_x, torch.sub), masked_operator, torch.sub)
        return nabla_f

    def _calculate_alphas(
            self, h, h_previous, nabla_f, nabla_f_previous, alpha_model=None):
        """Compute the coefficient in the steepest gradient method based on
        Barzilai-Borwein method.
        """
        if isinstance(h, list):
            if self.learn_alpha:
                list_alpha_model = [
                    self.alpha_model[k] for k in self.mask_function.mask.keys()
                    if k in self.alpha_model]
                return [
                    self._calculate_alphas(
                        h_, h_previous_, nabla_f_, nabla_f_previous_,
                        alpha_model=a)
                    for h_, h_previous_, nabla_f_, nabla_f_previous_, a
                    in zip(
                        h, h_previous, nabla_f, nabla_f_previous,
                        list_alpha_model)]
            else:
                return [
                    self._calculate_alphas(
                        h_, h_previous_, nabla_f_, nabla_f_previous_)
                    for h_, h_previous_, nabla_f_, nabla_f_previous_
                    in zip(h, h_previous, nabla_f, nabla_f_previous)]

        delta_h = h - h_previous
        delta_nabla_f = nabla_f - nabla_f_previous
        if not self.learn_alpha \
                and torch.allclose(
                    delta_h, torch.tensor(0.).to(delta_h.device)) \
                and torch.allclose(
                    delta_nabla_f, torch.tensor(0.).to(delta_nabla_f.device)):
            return 1.
        else:
            if self.learn_alpha:
                alpha = alpha_model(
                    torch.cat([
                        torch.einsum(
                            'n...f,n...f->f', delta_h, delta_h),
                        torch.einsum(
                            'n...f,n...f->f', delta_nabla_f, delta_nabla_f),
                        torch.einsum(
                            'n...f,n...f->f', delta_h, delta_nabla_f),
                    ], dim=-1))
            else:
                if self.componentwise_alpha:
                    if self.alpha_denominator_norm:
                        alpha = torch.einsum(
                            'n...f,n...f->f', delta_h, delta_nabla_f) / (
                                torch.einsum(
                                    'n...f,n...f->f',
                                    delta_nabla_f, delta_nabla_f)
                                + 1.e-5)
                    else:
                        alpha = torch.einsum(
                            'n...f,n...f->f', delta_h, delta_h) / (
                                torch.einsum(
                                    'n...f,n...f->f', delta_h, delta_nabla_f)
                                + 1.e-5)
                else:
                    if self.alpha_denominator_norm:
                        alpha = torch.einsum(
                            '...,...->', delta_h, delta_nabla_f) / (
                                torch.einsum(
                                    '...,...->',
                                    delta_nabla_f, delta_nabla_f) + 1.e-5)
                    else:
                        alpha = torch.einsum(
                            '...,...->', delta_h, delta_h) / (
                                torch.einsum(
                                    '...,...->', delta_h, delta_nabla_f)
                                + 1.e-5)
        return self.alpha_filter(alpha)

    def broadcast(self, a, x, operator):
        if isinstance(x, list):
            return [self.broadcast(a, x_, operator) for x_ in x]
        else:
            return operator(a, x)

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
            try:
                return {
                    k: torch.cat([
                        dict_predecessors[v_][k] for v_ in v
                        if k in dict_predecessors[v_]], dim=-1)
                    for k, v in self.input_names.items()}
            except:  # NOQA
                dict_shapes = {
                    k: [
                        dict_predecessors[v_][k].shape for v_ in v
                        if k in dict_predecessors[v_]]
                    for k, v in self.input_names.items()}
                raise ValueError(f"Input generation failed with:{dict_shapes}")

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

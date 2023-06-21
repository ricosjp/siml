import dataclasses as dc
import io
import os
import typing
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import yaml

from siml import util
from siml.path_like_objects import SimlFileBuilder


@dc.dataclass
class TypedDataClass:

    @classmethod
    def read_settings_yaml(cls, settings_yaml):
        settings_yaml = Path(settings_yaml)

        dict_settings = util.load_yaml_file(settings_yaml)
        return cls(**dict_settings)

    def convert(self):
        """Convert all fields accordingly with their type definitions."""
        for field_name, field in self.__dataclass_fields__.items():
            # self._convert_field(field_name, field)
            try:
                self._convert_field(field_name, field)
            except TypeError as e:
                raise TypeError(
                    f"{e}\n"
                    f"Can't convert {getattr(self, field_name)} to "
                    f"{typing.Type[field.type]} for {field_name}")

    def validate(self):
        for field_name, field in self.__dataclass_fields__.items():
            if not self._validate_field(field_name, field):
                raise TypeError(
                    f"{field_name} is not an instance of {field.type} "
                    f"(actual: {getattr(self, field_name).__class__})"
                )

    def _convert_field(self, field_name, field):
        if 'convert' in field.metadata and not field.metadata['convert']:
            return
        if 'allow_none' in field.metadata and field.metadata['allow_none'] \
                and getattr(self, field_name) is None:
            return

        def to_list_if_needed(x):
            if isinstance(x, np.ndarray):
                x = list(x)
            if not isinstance(x, (list, tuple)):
                x = [x]
            return x

        if field.type == list[Path]:
            def type_function(x):
                x = to_list_if_needed(x)
                return [Path(_x) for _x in x]
        elif field.type == list[str]:
            def type_function(x):
                x = to_list_if_needed(x)
                return [str(_x) for _x in x]
        elif field.type == list[int]:
            def type_function(x):
                x = to_list_if_needed(x)
                return [int(_x) for _x in x]
        elif field.type == list[bool]:
            def type_function(x):
                x = to_list_if_needed(x)
                return [bool(_x) for _x in x]
        elif field.type == list[float]:
            def type_function(x):
                x = to_list_if_needed(x)
                return [float(_x) for _x in x]
        elif field.type == list[dict]:
            def type_function(x):
                x = to_list_if_needed(x)
                return [dict(_x) for _x in x]
        elif field.type == typing.Tuple:
            def type_function(x):
                return tuple(_x for _x in x)
        elif field.type == slice:
            def type_function(x):
                if isinstance(x, slice):
                    return x
                else:
                    return slice(*x)
        elif field.type == CollectionVariableSetting:
            def type_function(x):
                if isinstance(x, dict):
                    if 'super_post_init' in x:
                        x['super_post_init'] = False
                        return CollectionVariableSetting(**x)
                    elif 'variables' in x:
                        return CollectionVariableSetting(
                            **x, super_post_init=False)
                    else:
                        return CollectionVariableSetting(
                            x, super_post_init=False)
                else:
                    return CollectionVariableSetting(x, super_post_init=False)
        elif field.type == typing.Union[
                list[VariableSetting],
                dict[str, list[VariableSetting]]]:
            def type_function(x):
                return CollectionVariableSetting(x, super_post_init=False)
        elif field.type == typing.Union[
                list[dict], dict[str, list]]:
            def type_function(x):
                if isinstance(x, list):
                    return [_x for _x in x]
                elif isinstance(x, dict):
                    return {key: list(value) for key, value in x.items()}
                else:
                    raise ValueError(f"Unexpected input: {x}")
        elif field.type == typing.Union[
                list[int], dict[str, list]]:
            def type_function(x):
                if isinstance(x, list):
                    return [int(_x) for _x in x]
                elif isinstance(x, dict):
                    return {key: list(value) for key, value in x.items()}
                else:
                    raise ValueError(f"Unexpected input: {x}")
        elif field.type == typing.Union[
                list[str], dict[str, list]]:
            def type_function(x):
                if isinstance(x, list):
                    return [str(_x) for _x in x]
                elif isinstance(x, dict):
                    return {key: list(value) for key, value in x.items()}
                else:
                    raise ValueError(f"Unexpected input: {x}")
        elif field.type == str:
            def type_function(x):
                if x is None:
                    return None
                else:
                    return str(x)
        elif field.type == typing.Union[str, dict]:
            def type_function(x):
                return x
        elif field.type == typing.Optional[typing.Union[str, Path]]:
            def type_function(x):
                return x
        else:
            type_function = field.type

        setattr(
            self, field_name, type_function(getattr(self, field_name)))

    def _validate_field(self, field_name, field):
        return isinstance(getattr(self, field_name), field.type)

    def __post_init__(self):
        self.convert()
        # self.validate()
        return

    def to_dict(self):
        dict_data = dc.asdict(self)
        standardized_dict_data = _standardize_data(dict_data)
        return standardized_dict_data


@dc.dataclass
class DataSetting(TypedDataClass):

    raw: list[Path] = dc.field(
        default_factory=lambda: [Path('data/raw')])
    interim: list[Path] = dc.field(
        default_factory=lambda: [Path('data/interim')])
    preprocessed: list[Path] = dc.field(
        default_factory=lambda: [Path('data/preprocessed')])
    inferred: list[Path] = dc.field(
        default_factory=lambda: [Path('data/inferred')])
    train: list[Path] = dc.field(
        default_factory=lambda: [Path('data/preprocessed/train')])
    validation: list[Path] = dc.field(
        default_factory=lambda: [])
    develop: list[Path] = dc.field(
        default_factory=lambda: [])
    test: list[Path] = dc.field(
        default_factory=lambda: [])
    pad: bool = False
    encrypt_key: bytes = dc.field(
        default=None, metadata={'allow_none': True})

    def __post_init__(self):
        if self.pad:
            raise ValueError(
                "pad = True option is deprecated. Set pad = False")
        super().__post_init__()

    @property
    def raw_root(self):
        return self._find_root(self.raw)

    @property
    def interim_root(self):
        return self._find_root(self.interim)

    @property
    def preprocessed_root(self):
        return self._find_root(self.preprocessed)

    @property
    def inferred_root(self):
        return self._find_root(self.inferred)

    def _find_root(self, paths):
        common_path = str(Path(os.path.commonprefix(paths))).rstrip('*')
        if '*' in common_path:
            raise ValueError(f"Invalid paths: {paths}")
        return Path(common_path)


@dc.dataclass
class DBSetting(TypedDataClass):
    servername: str = ''
    username: str = ''
    password: str = ''
    use_sqlite: bool = False


class Iter(Enum):
    SERIAL = 'serial'
    MULTIPROCESS = 'multiprocess'
    MULTITHREAD = 'multithread'


@dc.dataclass
class StudySetting(TypedDataClass):

    root_directory: Path = dc.field(
        default=None, metadata={'allow_none': True})
    type: str = 'learning_curve'
    relative_develop_size_linspace: typing.Tuple = dc.field(
        default_factory=lambda: (.2, 1., 5))
    n_fold: int = 10
    unit_error: str = '-'
    plot_validation: bool = False
    x_from_zero: bool = False
    y_from_zero: bool = False
    x_logscale: bool = False
    y_logscale: bool = False
    scale_loss: bool = False


@dc.dataclass
class VariableSetting(TypedDataClass):
    """
    name: str
        The name of the variable.
    dim: int
        The number of the feature of the variable.
        For higher tensor variables, it should be the dimension of the last
        index.
    shape: list[int]
        The shape of the tensor.
    skip: bool
        If True, skip the variable for loss computation or convergence
        computation.
    time_series: bool
        If True, regard it as a time series.
    time_slice: list[int]
        Slice for time series.
    """
    name: str = 'variable'
    dim: int = 1
    shape: list[int] = dc.field(default_factory=list)
    skip: bool = False
    time_series: bool = False
    time_slice: slice = dc.field(default_factory=lambda: slice(None))

    def get(self, key, default=None):
        if default is None:
            return getattr(self, key)
        else:
            return getattr(self, key, default)


@dc.dataclass
class CollectionVariableSetting(TypedDataClass):

    variables: typing.Union[
        list[VariableSetting], dict[str, list[VariableSetting]]] = dc.field(
            default_factory=list)
    super_post_init: bool = True

    def __post_init__(self):
        self.strip()

        if self.variables is None or len(self.variables) == 0:
            self.variables = []
        elif isinstance(self.variables, dict):
            if 'variables' in self.variables:
                self.variables = CollectionVariableSetting(
                    self.variables['variables']).variables
            else:
                self.variables = {
                    key: CollectionVariableSetting(value)
                    for key, value in self.variables.items()}
        elif isinstance(self.variables, list):
            self.variables = [
                VariableSetting(**v) if isinstance(v, dict) else v
                for v in self.variables]
        else:
            raise ValueError(f"Unexpected variables: {self.variables}")

        if self.super_post_init:
            super().__post_init__()
        self.strip()
        return

    def __len__(self):
        return len(self.variables)

    def __getitem__(self, key):
        return self.variables[key]

    def __setitem__(self, key, value):
        self.variables[key] = value
        return

    def strip(self):
        while isinstance(self.variables, CollectionVariableSetting):
            self.variables = self.variables.variables
        return

    def collect_values(self, key, *, default=None):
        data = self.variables
        if isinstance(data, list):
            return [d.get(key) for d in data]
        elif isinstance(data, dict):
            return {
                dict_key: dict_value.collect_values(key, default=default)
                for dict_key, dict_value in data.items()}
        else:
            raise ValueError(f"Unexpected data: {data}")

    @property
    def is_dict(self):
        return isinstance(self.variables, dict)

    @property
    def names(self):
        return self.collect_values('name')

    @property
    def dims(self):
        return self.collect_values('dim', default=1)

    @property
    def length(self):
        dim_data = self.dims
        if isinstance(dim_data, dict):
            return {key: np.sum(value) for key, value in dim_data.items()}
        else:
            return np.sum(dim_data)

    @property
    def time_series(self):
        return self.collect_values('time_series')

    @property
    def time_slice(self):
        s = self.collect_values('time_slice')
        if isinstance(s, list):
            return s[0]
        elif isinstance(s, dict):
            return {k: v[0] for k, v in s.items()}
        else:
            raise ValueError(f"Invalid format: {s}")

    def get_time_series_keys(self):
        if not isinstance(self.time_series, dict):
            return []
        return [
            k for k, v in self.time_series.items() if np.any(v)
        ]

    def to_dict(self):
        if isinstance(self.variables, dict):
            ret_dict = {}
            for value in self.variables.values():
                ret_dict.update(value.to_dict())
            return ret_dict
        elif isinstance(self.variables, list):
            return {d.name: d.to_dict() for d in self.variables}
        else:
            raise ValueError(f"Unexpected self.variables: {self.variables}")


@dc.dataclass
class OptimizerSetting(TypedDataClass):
    lr: float = 0.001
    betas: typing.Tuple = \
        dc.field(default=(0.9, 0.999))
    eps: float = 1e-8
    weight_decay: float = 0


@dc.dataclass
class TrainerSetting(TypedDataClass):

    """
    inputs: siml.setting.CollectionVariableSetting
        Variable settings of inputs.
    outputs: siml.setting.CollectionVariableSetting
        Variable settings of outputs.
    train_directories: list[str] or pathlib.Path
        Training data directories.
    output_directory_base: str or pathlib.Path
        Output directory base name.
    output_directory: str or pathlib.Path
        Output directory name.
    validation_directories: list[str] or pathlib.Path, optional
        Validation data directories.
    restart_directory: str or pathlib.Path, optional
        Directory name to be used for restarting.
    pretrain_directory: str or pathlib.Path, optional
        Pretrained directory name.
    loss_function: chainer.FunctionNode,
            optional
        Loss function to be used for training.
    optimizer: chainer.Optimizer, optional
        Optimizer to be used for training.
    compute_accuracy: bool, optional
        If True, compute accuracy.
    name: str
        The name of the study.
    suffix: str
        Suffix to be added to the name.
    batch_size: int, optional
        Batch size for train dataset.
    validation_batch_size: int, optional
        Batch size for validation dataset.
    n_epoch: int, optional
        The number of epochs.
    model_key: bytes
        If fed, decrypt model file with the key.
    gpu_id: int, optional
        GPU ID. Specify non negative value to use GPU. -1 Meaning CPU.
    log_trigger_epoch: int, optional
        The interval of logging of training. It is used for logging,
        plotting, and saving snapshots.
    stop_trigger_epoch: int, optional
        The interval to check if training should be stopped. It is used
        for early stopping and pruning.
    optuna_trial: optuna.Trial, optional
        Trial object used to perform optuna hyper parameter tuning.
    prune: bool, optional
        If True and optuna_trial is given, prining would be performed.
    seed: str, optional
        Random seed.
    element_wise: bool, optional
        If True, concatenate data to force element wise training
        (so no graph information can be used). With this option,
        element_batch_size will be used for trainer's batch size as it is
        "element wise" training.
    element_batch_size: int, optional
        If positive, split one mesh int element_batch_size and perform update
        multiple times for one mesh. In case of element_wise is True,
        element_batch_size is the batch size in the usual sence.
    validation_element_batch_size: int, optional
        element_batch_size for validation dataset.
    simplified_model: bool, optional
        If True, regard the target simulation as simplified simulation
        (so-called "1D simulation"), which focuses on only a few inputs and
        outputs. The behavior of the trainer will be similar to that with
        element_wise = True.
    time_series: bool, optional
        If True, regard the data as time series. In that case, the data shape
        will be [seq, batch, element, feature] instead of the default
        [batch, element, feature] shape.
    lazy: bool, optional
        If True, load data lazily.
    num_workers: int, optional
        The number of workers to load data.
    display_mergin: int, optional
    non_blocking: bool [True]
        If True and this copy is between CPU and GPU, the copy may occur
        asynchronously with respect to the host. For other cases, this argument
        has no effect.
    data_parallel: bool [False]
        If True, perform data parallel on GPUs.
    model_parallel: bool [False]
        If True, perform model parallel on GPUs.
    draw_network: bool [True]
        If True, draw network (requireing graphviz).
    output_stats: bool [False]
        If True, output stats of training (like mean of weight, grads, ...)
    split_ratio: dict[str, float]
        If fed, split the data into train, validation, and test at the
        beginning of the training. Should be
        {'validation': float, 'test': float} dict.
    figure_format: str
        The format of the figure. The default is 'pdf'.
    clip_grad_value: float
        If fed, apply gradient clipping by value.
    clip_grad_norm: float
        If fed, apply gradient clipping with norm.
    recursive: bool
        If True, search data recursively.
    time_series_split: list[int]
        If fed, split time series with [start, step, length].
    loss_slice: slice
        Slice to be applied to loss computation.
    state_dict_strict: bool
        It will be passed to torch.nn.Module.load_state_dict.
    """

    inputs: CollectionVariableSetting = dc.field(
        default_factory=CollectionVariableSetting)
    support_input: str = dc.field(default=None, metadata={'allow_none': True})
    support_inputs: list[str] = dc.field(
        default=None, metadata={'allow_none': True})
    outputs: CollectionVariableSetting = dc.field(
        default_factory=CollectionVariableSetting)
    output_directory_base: Path = Path('models')
    output_directory: Path = None

    name: str = 'default'
    suffix: str = dc.field(
        default=None, metadata={'allow_none': True})
    batch_size: int = 1
    validation_batch_size: int = dc.field(
        default=None, metadata={'allow_none': True})
    n_epoch: int = 100

    validation_directories: list[Path] = dc.field(
        default_factory=lambda: [])
    restart_directory: Path = dc.field(
        default=None, metadata={'allow_none': True})
    pretrain_directory: Path = dc.field(
        default=None, metadata={'allow_none': True})
    loss_function: typing.Union[str, dict] = 'mse'
    optimizer: str = 'adam'
    compute_accuracy: bool = False
    model_key: bytes = dc.field(
        default=None, metadata={'allow_none': True})
    gpu_id: int = -1
    log_trigger_epoch: int = 1
    stop_trigger_epoch: int = 10
    patience: int = 3
    optuna_trial: optuna.Trial = dc.field(
        default=None, metadata={'convert': False, 'allow_none': True})
    prune: bool = False
    snapshot_choise_method: str = 'best'
    seed: int = 0
    element_wise: bool = False
    simplified_model: bool = False
    time_series: bool = False
    element_batch_size: int = -1
    validation_element_batch_size: int = dc.field(
        default=None, metadata={'allow_none': True})
    use_siml_updater: bool = True
    iterator: Iter = Iter.SERIAL
    optimizer_setting: dict = dc.field(default_factory=dict)
    lazy: bool = True
    num_workers: int = dc.field(
        default=None, metadata={'allow_none': True})
    display_mergin: int = 4
    non_blocking: bool = True
    clip_grad_value: float = dc.field(
        default=None, metadata={'allow_none': True})
    clip_grad_norm: float = dc.field(
        default=None, metadata={'allow_none': True})
    recursive: bool = True
    state_dict_strict: bool = True

    data_parallel: bool = False
    model_parallel: bool = False
    draw_network: bool = True
    output_stats: bool = False
    split_ratio: dict = dc.field(default_factory=dict)
    figure_format: str = 'pdf'

    pseudo_batch_size: int = 0
    time_series_split: list[int] = dc.field(
        default=None, metadata={'allow_none': True})
    time_series_split_evaluation: list[int] = dc.field(
        default=None, metadata={'allow_none': True})
    loss_slice: slice = dc.field(default_factory=lambda: slice(None))

    def __post_init__(self):
        if self.element_wise and self.lazy:
            self.lazy = False
            print('element_wise = True found. Overwrite lazy = False.')
        if self.simplified_model and self.lazy:
            raise ValueError(
                'Both simplified_model and lazy cannot be True '
                'at the same time')

        if self.validation_batch_size is None:
            self.validation_batch_size = self.batch_size

        if self.validation_element_batch_size is None:
            self.validation_element_batch_size = self.element_batch_size

        if self.output_directory is None:
            self.update_output_directory()
        if self.support_input is not None:
            if self.support_inputs is not None:
                if len(self.support_inputs) == 1 \
                        and self.support_inputs[0] == self.support_input:
                    pass
                else:
                    raise ValueError(
                        'support_input and support_inputs are set '
                        'inconsistently.')
            else:
                self.support_inputs = [self.support_input]

        self.optimizer_setting = \
            dc.asdict(OptimizerSetting(**self.optimizer_setting))

        if self.element_wise or self.simplified_model:
            self.use_siml_updater = False

        if self.num_workers is None:
            self.num_workers = util.determine_max_process()

        if (self.stop_trigger_epoch // self.log_trigger_epoch) == 0:
            raise ValueError(
                "Set stop_trigger_epoch larger than log_trigger_epoch")

        if self.time_series_split_evaluation is None:
            self.time_series_split_evaluation = self.time_series_split
        if self.time_series:
            self.update_time_series(self.inputs)
            self.update_time_series(self.outputs)
        super().__post_init__()
        return

    def update_time_series(self, variables):
        if isinstance(variables, list):
            for variable in variables:
                self.update_time_series(variable)
        elif isinstance(variables, dict):
            if 'super_post_init' in variables:
                self.update_time_series(variables['variables'])
            elif 'name' in variables:
                variables['time_series'] = True
            else:
                for variable in variables.values():
                    self.update_time_series(variable)
        else:
            raise ValueError(f"Unexpected variables type: {variables}")
        return

    @property
    def output_skips(self):
        return self.outputs.collect_values(
            'skip', default=False)

    @property
    def input_names(self):
        return self.inputs.names

    @property
    def input_dims(self):
        return self.inputs.dims

    @property
    def output_names(self):
        return self.outputs.names

    @property
    def output_dims(self):
        return self.outputs.dims

    @property
    def input_length(self):
        return self.inputs.length

    @property
    def output_length(self):
        return self.outputs.length

    @property
    def variable_information(self):
        out_dict = self.inputs.to_dict()
        out_dict.update(self.outputs.to_dict())
        return out_dict

    def determine_element_wise(self) -> bool:
        if self.time_series:
            return False
        if self.element_wise or self.simplified_model:
            return True

        return False

    def update_output_directory(self, *, id_=None, base=None):
        if base is None:
            base = Path(self.output_directory_base)
        else:
            base = Path(base)
        if id_ is None:
            id_string = ''
        else:
            id_string = f"_{id_}"
        if self.suffix is None:
            suffix_string = ''
        else:
            suffix_string = f"_{self.suffix}"
        self.output_directory = base \
            / f"{self.name}{suffix_string}{id_string}_{util.date_string()}"


@dc.dataclass
class InfererSetting(TypedDataClass):
    """
    model: pathlib.Path optional
        Model directory, file path, or buffer. If not fed,
        TrainerSetting.pretrain_directory will be used.
    save: bool, optional
        If True, save inference results.
    output_directory: pathlib.Path, optional
        Output directory path. If fed, output the data in the specified
        directory. When this is fed, output_directory_base has no effect.
    output_directory_base: pathlib.Path, optional
        Output directory base name. If not fed, data/inferred will be the
        default output directory base.
    data_directories: list[pathlib.Path], optional
        Data directories to infer.
    write_simulation: bool, optional
        If True, write simulation data file(s) based on the inference.
    write_npy: bool, optional
        If True, write npy files of inferences.
    write_yaml: bool, optional
        If True, write yaml file used to make inference.
    write_simulation_base: pathlib.Path, optional
        Base of simulation data to be used for write_simulation option.
        If not fed, try to find from the input directories.
    read_simulation_type: str, optional
        Simulation file type to read.
    write_simulation_type: str, optional
        Simulation file type to write.
    converter_parameters_pkl: pathlib.Path, optional
        Pickel file of converter parameters. IF not fed,
        DataSetting.preprocessed_root is used.
    perform_preprocess: bool, optional
        If True, perform preprocess.
    accomodate_length: int
        If specified, duplicate initial state to initialize RNN state.
    overwrite: bool
        If True, overwrite output.
    return_all_results: bool
        If True, return all inference results. Set False if the inference data
        is too large to fit into the memory available.
    model_key: bytes
        If fed, decrypt model file with the key.
    gpu_id: int, optional
        GPU ID. Specify non negative value to use GPU. -1 Meaning CPU.
    less_output: bool, optional
        If True, output less variables in FEMData object.
    skip_fem_data_creation: bool, optional
        If True, skip fem_data object creation.
    """
    model: Path = dc.field(
        default=None, metadata={'allow_none': True})
    save: bool = True
    overwrite: bool = False
    output_directory: Path = dc.field(
        default=None, metadata={'allow_none': True})
    output_directory_base: Path = Path('data/inferred')
    overwrite: bool = False
    data_directories: list[Path] = dc.field(
        default_factory=list)
    write_simulation: bool = False
    write_npy: bool = True
    write_yaml: bool = True
    write_simulation_base: Path = dc.field(
        default=None, metadata={'allow_none': True})
    write_simulation_stem: Path = dc.field(
        default=None, metadata={'allow_none': True})
    read_simulation_type: str = 'fistr'
    write_simulation_type: str = 'fistr'
    converter_parameters_pkl: Path = dc.field(
        default=None, metadata={'allow_none': True})
    convert_to_order1: bool = False
    accomodate_length: int = 0
    perform_preprocess: bool = False
    perform_inverse: bool = True
    return_all_results: bool = True
    model_key: bytes = dc.field(
        default=None, metadata={'allow_none': True})
    gpu_id: int = -1
    less_output: bool = False
    skip_fem_data_creation: bool = False
    infer_epoch: int = dc.field(
        default=None, metadata={'allow_none': True})


@dc.dataclass
class BlockSetting(TypedDataClass):
    name: str = 'Block'
    is_first: bool = False
    is_last: bool = False
    type: str = dc.field(
        default=None, metadata={'allow_none': True})
    destinations: list[str] = dc.field(
        default_factory=list)
    residual: bool = False
    reference_block_name: str = None
    activation_after_residual: bool = True
    allow_linear_residual: bool = False
    bias: bool = True
    input_slice: slice = slice(0, None, 1)
    input_indices: list[int] = dc.field(
        default=None, metadata={'allow_none': True})
    input_keys: list[str] = dc.field(
        default=None, metadata={'allow_none': True})
    input_names: list[str] = dc.field(
        default=None, metadata={'allow_none': True})
    output_key: str = dc.field(
        default=None, metadata={'allow_none': True})
    support_input_index: int = dc.field(
        default=None, metadata={'allow_none': True})
    support_input_indices: list[int] = dc.field(
        default=None, metadata={'allow_none': True})
    nodes: list[int] = dc.field(
        default_factory=lambda: [-1, -1])
    kernel_sizes: list[int] = dc.field(
        default=None, metadata={'allow_none': True})
    activations: list[str] = dc.field(
        default_factory=lambda: ['identity'])
    dropouts: list[float] = dc.field(
        default=None, metadata={'allow_none': True})
    device: int = dc.field(
        default=None, metadata={'allow_none': True})
    coeff: float = dc.field(
        default=None, metadata={'allow_none': True})
    time_series: bool = False
    no_grad: bool = False
    weight_norm: bool = False
    losses: list[dict] = dc.field(default_factory=list)
    clip_grad_value: float = dc.field(
        default=None, metadata={'allow_none': True})
    clip_grad_norm: float = dc.field(
        default=None, metadata={'allow_none': True})

    optional: dict = dc.field(default_factory=dict)

    # Parameters for dynamic definition of layers
    hidden_nodes: int = dc.field(
        default=None, metadata={'allow_none': True})
    hidden_layers: int = dc.field(
        default=None, metadata={'allow_none': True})
    hidden_activation: str = 'relu'
    output_activation: str = 'identity'
    input_dropout: float = 0.0
    hidden_dropout: float = 0.0
    output_dropout: float = 0.0

    def __post_init__(self):
        if self.dropouts is None:
            self.dropouts = [0] * (len(self.nodes) - 1)

        # Dynamic definition of layers
        if self.hidden_nodes is not None and self.hidden_layers is not None:
            self.nodes = \
                [-1] + [self.hidden_nodes] * self.hidden_layers + [-1]
            self.activations = [self.hidden_activation] * self.hidden_layers \
                + [self.output_activation]
            self.dropouts = [self.input_dropout] \
                + [self.hidden_dropout] * (self.hidden_layers - 1) \
                + [self.output_dropout]
        if len(self.activations) != len(self.nodes) - 1:
            raise ValueError(
                f"len(activations) should be {len(self.nodes)-1} but "
                f"{len(self.activations)} for {self}")
        if len(self.dropouts) != len(self.nodes) - 1:
            raise ValueError(
                f"len(dropouts) should be {len(self.nodes)-1} but "
                f"{len(self.dropouts)} for {self}")
        if self.kernel_sizes is not None:
            if len(self.kernel_sizes) != len(self.nodes) - 1:
                raise ValueError(
                    f"len(kernel_sizes) should be {len(self.nodes)-1} but "
                    f"{len(self.kernel_sizes)} for {self}")

        for i, _ in enumerate(self.losses):
            if isinstance(self.losses[i], str):
                self.losses[i] = {'name': self.losses[i], 'coeff': 1.}
        super().__post_init__()

        if self.input_indices is not None:
            self.input_selection = self.input_indices
        else:
            self.input_selection = self.input_slice

        if self.support_input_indices is None:
            if self.support_input_index is None:
                self.support_input_indices = [0]
            else:
                self.support_input_indices = [self.support_input_index]

        return

    @property
    def loss_names(self):
        return [loss_setting['name'] for loss_setting in self.losses]


@dc.dataclass
class GroupSetting(TypedDataClass):
    blocks: list[BlockSetting]
    name: str = 'GROUP'
    inputs: CollectionVariableSetting = dc.field(
        default_factory=CollectionVariableSetting)
    support_inputs: list[str] = dc.field(
        default=None, metadata={'allow_none': True})
    outputs: CollectionVariableSetting = dc.field(
        default_factory=CollectionVariableSetting)
    repeat: int = 1
    convergence_threshold: float = dc.field(
        default=None, metadata={'allow_none': True})
    mode: str = 'simple'
    debug: bool = False
    time_series_length: int = dc.field(
        default=None, metadata={'allow_none': True})

    optional: dict = dc.field(default_factory=dict)

    def __post_init__(self):
        self.blocks = [
            self._convert_block_if_needed(block) for block in self.blocks]
        super().__post_init__()
        return

    def _convert_block_if_needed(self, block):
        if isinstance(block, BlockSetting):
            return block
        elif isinstance(block, dict):
            return BlockSetting(**block)
        else:
            raise ValueError(f"Unexpected block type: {block}")

    @property
    def input_names(self):
        return self.inputs.names

    @property
    def input_dims(self):
        return self.inputs.dims

    @property
    def output_names(self):
        return self.outputs.names

    @property
    def output_dims(self):
        return self.outputs.dims

    @property
    def input_length(self):
        return self.inputs.length

    @property
    def output_length(self):
        return self.outputs.length


@dc.dataclass
class ModelSetting(TypedDataClass):
    blocks: list[BlockSetting]
    groups: list[GroupSetting] = None

    def __init__(self, setting=None, blocks=None, groups=None):
        if groups is not None:
            self.groups = groups

        if blocks is not None:
            self.blocks = blocks
        else:
            if setting is None:
                self.blocks = [BlockSetting()]
            else:
                self.blocks = [
                    BlockSetting(**block) for block in setting['blocks']]
                if 'groups' in setting:
                    if setting['groups'] is None:
                        if groups is None:
                            self.groups = []
                    else:
                        if groups is None:
                            self.groups = [
                                GroupSetting(**group)
                                for group in setting['groups']]
                        else:
                            raise ValueError(
                                f"groups found in the setting for: {setting}")
                else:
                    if groups is None:
                        self.groups = []

        if np.all(b.is_first is False for b in self.blocks):
            self.blocks[0].is_first = True
        if np.all(b.is_last is False for b in self.blocks):
            self.blocks[-1].is_last = True

        return


@dc.dataclass
class OptunaSetting(TypedDataClass):
    n_trial: int = 100
    output_base_directory: Path = Path('models/optuna')
    hyperparameters: list[dict] = dc.field(default_factory=list)
    setting: dict = dc.field(default_factory=dict)

    def __post_init__(self):
        for hyperparameter in self.hyperparameters:
            if hyperparameter['type'] == 'categorical':
                if len(hyperparameter['choices']) != len(np.unique([
                        c['id'] for c in hyperparameter['choices']])):
                    raise ValueError(
                        'IDs in optuna.hyperparameter.choices not unique')
        super().__post_init__()


@dc.dataclass
class ConversionSetting(TypedDataClass):
    """Dataclass for raw data converter.

    Parameters
    -----------
    mandatory_variables: list[str]
        Mandatory variable names. If any of them are not found,
        ValueError is raised.
    mandatory: list[str]
        An alias of mandatory_variables.
    optional_variables: list[str]
        Optional variable names. If any of them are not found,
        they are ignored.
    optional: list[str]
        An alias of optional_variables.
    output_base_directory: str or pathlib.Path, optional
        Output base directory for the converted raw data. By default,
        'data/interim' is the output base directory, so
        'data/interim/aaa/bbb' directory is the output directory for
        'data/raw/aaa/bbb' directory.
    finished_file: str, optional
        File name to indicate that the conversion is finished.
    file_type: str, optional
        File type to be read.
    required_file_names: list[str], optional
        Required file names.
    skip_femio: bool, optional
        If True, skip femio.FEMData reading process. Useful for
        user-defined data format such as csv, h5, etc.
    time_series: bool, optional
        If True, make femio parse time series data.
    save_femio: bool, optional
        If True, save femio data in the interim directories.
    skip_save: bool, optional
        If True, skip SiML's default saving function.
    max_process: int, optional
        Maximum number of processes.
    """

    mandatory_variables: list[str] = dc.field(
        default_factory=list)
    optional_variables: list[str] = dc.field(
        default_factory=list)
    mandatory: list[str] = dc.field(
        default_factory=list)
    optional: list[str] = dc.field(
        default_factory=list)
    output_base_directory: typing.Optional[typing.Union[Path, str]] = None
    finished_file: str = 'converted'
    file_type: str = 'fistr'
    required_file_names: list[str] = dc.field(default_factory=list)
    skip_femio: bool = False
    time_series: bool = False
    save_femio: bool = False
    skip_save: bool = False
    max_process: int = 1000

    @classmethod
    def read_settings_yaml(cls, settings_yaml):
        dict_settings = util.load_yaml_file(settings_yaml)
        data = DataSetting(**dict_settings['data'])
        return cls(**dict_settings['raw_conversion'], data=data)

    def __post_init__(self):
        if len(self.mandatory) > len(self.mandatory_variables):
            self.mandatory_variables = self.mandatory
        elif len(self.mandatory) < len(self.mandatory_variables):
            self.mandatory = self.mandatory_variables
        else:
            pass

        if len(self.optional) > len(self.optional_variables):
            self.optional_variables = self.optional
        elif len(self.optional) < len(self.optional_variables):
            self.optional = self.optional_variables
        else:
            pass

        super().__post_init__()
        return

    @property
    def should_load_mandatory_variables(self) -> bool:
        if self.mandatory_variables is None:
            return False

        return len(self.mandatory_variables) > 0


@dc.dataclass
class PreprocessSetting:
    preprocess: dict = dc.field(default_factory=dict)

    @classmethod
    def read_settings_yaml(cls, settings_yaml):
        dict_settings = util.load_yaml_file(settings_yaml)
        preprocess = dict_settings['preprocess']
        return cls(preprocess=preprocess)

    def __post_init__(self):
        for key, value in self.preprocess.items():
            if isinstance(value, str):
                self.preprocess.update({key: {
                    'method': value, 'componentwise': False, 'same_as': None,
                    'group_id': 0, 'power': 1., 'other_components': []}})
            elif isinstance(value, dict):
                if 'method' not in value:
                    value.update({'method': 'identity'})
                if 'componentwise' not in value:
                    value.update({'componentwise': False})
                if 'same_as' not in value:
                    value.update({'same_as': None})
                if 'group_id' not in value:
                    value.update({'group_id': 0})
                if 'power' not in value:
                    value.update({'power': 1.})
                if 'other_components' not in value:
                    value.update({'other_components': []})
                self.preprocess.update({key: value})
            else:
                raise ValueError(
                    f"Invalid format of preprocess setting: {self}")
        return


@dc.dataclass
class MainSetting:
    data: DataSetting = dc.field(default_factory=DataSetting)
    conversion: ConversionSetting = dc.field(default_factory=ConversionSetting)
    preprocess: dict = dc.field(default_factory=dict)
    trainer: TrainerSetting = dc.field(default_factory=TrainerSetting)
    inferer: InfererSetting = dc.field(default_factory=InfererSetting)
    model: ModelSetting = dc.field(default_factory=ModelSetting)
    optuna: OptunaSetting = dc.field(default_factory=OptunaSetting)
    study: StudySetting = dc.field(default_factory=StudySetting)
    replace_preprocessed: bool = False
    misc: dict = dc.field(default_factory=dict)

    @classmethod
    def read_settings_yaml(
        cls,
        settings_yaml: Path,
        replace_preprocessed=False,
        *,
        decrypt_key: Optional[bytes] = None
    ):

        siml_file = SimlFileBuilder.yaml_file(settings_yaml)
        dict_settings = siml_file.load(decrypt_key=decrypt_key)
        if not siml_file.is_encrypted:
            name = siml_file.file_path.stem
        else:
            name = None

        return cls.read_dict_settings(
            dict_settings,
            name=name,
            replace_preprocessed=replace_preprocessed
        )

    @classmethod
    def read_dict_settings(
            cls, dict_settings, *, name=None, replace_preprocessed=False):
        if 'trainer' in dict_settings \
                and 'name' not in dict_settings['trainer']:
            if name is None:
                dict_settings['trainer']['name'] = 'unnamed'
            else:
                dict_settings['trainer']['name'] = name
        if 'inferer' in dict_settings:
            inferer_setting = InfererSetting(**dict_settings['inferer'])
        else:
            inferer_setting = InfererSetting()
        if 'data' in dict_settings:
            data_setting = DataSetting(**dict_settings['data'])
        else:
            data_setting = DataSetting()
        if 'conversion' in dict_settings:
            conversion_setting = ConversionSetting(
                **dict_settings['conversion'])
        else:
            conversion_setting = ConversionSetting()
        if 'preprocess' in dict_settings:
            preprocess_setting = PreprocessSetting(
                dict_settings['preprocess']).preprocess
        else:
            preprocess_setting = PreprocessSetting().preprocess
        if 'trainer' in dict_settings:
            trainer_settings = cls._pop_train_setting(dict_settings['trainer'])
            trainer_setting = TrainerSetting(**trainer_settings)
        else:
            trainer_setting = TrainerSetting()
        if 'model' in dict_settings:
            model_setting = ModelSetting(dict_settings['model'])
        else:
            model_setting = ModelSetting()
        if 'optuna' in dict_settings:
            optuna_setting = OptunaSetting(**dict_settings['optuna'])
        else:
            optuna_setting = OptunaSetting()
        if 'study' in dict_settings:
            study_setting = StudySetting(**dict_settings['study'])
        else:
            study_setting = StudySetting()
        if 'misc' in dict_settings:
            misc_setting = dict_settings['misc']
        else:
            misc_setting = {}

        return cls(
            data=data_setting, conversion=conversion_setting,
            preprocess=preprocess_setting,
            trainer=trainer_setting, model=model_setting,
            inferer=inferer_setting,
            optuna=optuna_setting, study=study_setting,
            replace_preprocessed=replace_preprocessed,
            misc=misc_setting,
        )

    @classmethod
    def _pop_train_setting(cls, train_dict_settings):
        # Pop unnecessary settings for backward compatibility
        train_dict_settings.pop('input_names', None)
        train_dict_settings.pop('input_dims', None)
        train_dict_settings.pop('input_length', None)
        train_dict_settings.pop('output_names', None)
        train_dict_settings.pop('output_dims', None)
        train_dict_settings.pop('output_length', None)
        return train_dict_settings

    def __post_init__(self):

        for block in self.model.blocks:
            block.time_series = self.trainer.time_series

        if self.replace_preprocessed:
            if str(self.data.preprocessed[0]) \
                    != str(self.data.train[0].parent) \
                    and str(self.data.preprocessed[0]) \
                    != str(self.data.train[0]):
                print(
                    'self.data.preprocessed differs from self.data.train. '
                    'Replaced.')
                self.data.preprocessed = [self.data.train[0].parent]
        return

    def get_crypt_key(self):
        if self.data.encrypt_key is not None:
            return self.data.encrypt_key

        if self.inferer.model_key is not None:
            return self.inferer.model_key

        if self.trainer.model_key is not None:
            return self.trainer.model_key

        return None

    def update_with_dict(self, new_dict):
        original_dict = dc.asdict(self)
        return MainSetting.read_dict_settings(
            self._update_with_dict(original_dict, new_dict))

    def _update_with_dict(self, original_setting, new_setting):
        if isinstance(new_setting, str) or isinstance(new_setting, float) \
                or isinstance(new_setting, int):
            return new_setting
        elif isinstance(new_setting, list):
            if 'variables' in original_setting:
                # For backward compatibility
                original_setting['variables'] += new_setting
                return original_setting
            else:
                return [
                    self._update_with_dict(original_setting, s)
                    for s in new_setting]
        elif isinstance(new_setting, dict):
            for key, value in new_setting.items():
                if isinstance(original_setting, list):
                    return new_setting
                original_setting.update({
                    key: self._update_with_dict(original_setting[key], value)})
            return original_setting
        elif isinstance(new_setting, Path):
            return str(new_setting)
        elif isinstance(new_setting, np.ndarray):
            return self._update_with_dict(
                original_setting, new_setting.tolist())
        else:
            raise ValueError(f"Unknown data type: {new_setting.__class__}")


def write_yaml(data_class, file_name, *, overwrite=False, key=None):
    """Write YAML file of the specified dataclass object.

    Parameters
    -----------
    data_class: dataclasses.dataclass
        DataClass object to write.
    file_name: str or pathlib.Path
        YAML file name to write.
    overwrite: bool, optional
        If True, overwrite file.
    key: bytes
        Key for encription.
    """
    if key is None:
        file_name = Path(file_name)
    else:
        if '.enc' in str(file_name):
            file_name = Path(file_name)
        else:
            file_name = Path(str(file_name) + '.enc')
    if file_name.exists() and not overwrite:
        raise ValueError(f"{file_name} already exists")

    if key is None:
        with open(file_name, 'w') as f:
            dump_yaml(data_class, f)
    else:
        string = dump_yaml(data_class, None)
        bio = io.BytesIO(string.encode('utf8'))
        util.encrypt_file(key, file_name, bio)
    return


def dump_yaml(data_class, stream):
    """Write YAML file of the specified dataclass object.

    Parameters
    -----------
    data_class: dataclasses.dataclass
        DataClass object to write.
    stream: File or stream
        Stream to write.
    """
    dict_data = dc.asdict(data_class)
    standardized_dict_data = _standardize_data(dict_data)

    # Remove keys
    if 'encrypt_key' in standardized_dict_data:
        standardized_dict_data.pop('encrypt_key')
    if 'decrypt_key' in standardized_dict_data:
        standardized_dict_data.pop('decrypt_key')
    if 'model_key' in standardized_dict_data:
        standardized_dict_data.pop('model_key')
    if 'data' in standardized_dict_data:
        if 'encrypt_key' in standardized_dict_data['data']:
            standardized_dict_data['data'].pop('encrypt_key')
        if 'decrypt_key' in standardized_dict_data['data']:
            standardized_dict_data['data'].pop('decrypt_key')
    if 'trainer' in standardized_dict_data:
        if 'model_key' in standardized_dict_data['trainer']:
            standardized_dict_data['trainer'].pop('model_key')
    if 'inferer' in standardized_dict_data:
        if 'model_key' in standardized_dict_data['inferer']:
            standardized_dict_data['inferer'].pop('model_key')

    return yaml.dump(standardized_dict_data, stream)


def _standardize_data(data):
    if isinstance(data, list):
        return [_standardize_data(d) for d in data]
    elif isinstance(data, np.ndarray):
        return [_standardize_data(d) for d in data]
    elif isinstance(data, tuple):
        return [_standardize_data(d) for d in data]
    elif isinstance(data, slice):
        return [data.start, data.stop, data.step]
    elif isinstance(data, dict):
        return {k: _standardize_data(v) for k, v in data.items()}
    elif isinstance(data, Path):
        return str(data)
    elif isinstance(data, Enum):
        return data.value
    else:
        return data

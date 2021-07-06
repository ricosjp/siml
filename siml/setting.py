import dataclasses as dc
from enum import Enum
import os
from pathlib import Path
import typing

import numpy as np
import optuna
import yaml

from . import util


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
            try:
                self._convert_field(field_name, field)
            except TypeError:
                raise TypeError(
                    f"Can't convert {getattr(self, field_name)} to "
                    f"{field.type} for {field_name}")

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
        elif field.type == typing.Union[
                list[dict], dict[str, list]]:
            def type_function(x):
                if isinstance(x, list):
                    return [dict(_x) for _x in x]
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

        return

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
class TrainerSetting(TypedDataClass):

    """
    inputs: list[dict] or dict
        Variable names of inputs.
    outputs: list[dict] or dict
        Variable names of outputs.
    train_directories: list[str] or pathlib.Path
        Training data directories.
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
    batch_size: int, optional
        Batch size for train dataset.
    validation_batch_size: int, optional
        Batch size for validation dataset.
    n_epoch: int, optional
        The number of epochs.
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
    """

    inputs: typing.Union[list[dict], dict[str, list]] \
        = dc.field(default_factory=list)
    support_input: str = dc.field(default=None, metadata={'allow_none': True})
    support_inputs: list[str] = dc.field(
        default=None, metadata={'allow_none': True})
    outputs: typing.Union[list[dict], dict[str, list]] \
        = dc.field(default_factory=list)
    output_directory: Path = None

    name: str = 'default'
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
    loss_function: str = 'mse'
    optimizer: str = 'adam'
    compute_accuracy: bool = False
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
    optimizer_setting: dict = dc.field(
        default=None, metadata={'convert': False, 'allow_none': True})
    lazy: bool = True
    num_workers: int = dc.field(
        default=None, metadata={'allow_none': True})
    display_mergin: int = 5
    non_blocking: bool = True

    data_parallel: bool = False
    model_parallel: bool = False
    draw_network: bool = True
    output_stats: bool = False
    split_ratio: dict = dc.field(default_factory=dict)
    figure_format: str = 'pdf'

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
        if self.optimizer_setting is None:
            self.optimizer_setting = {
                'lr': 0.001,
                'betas': (0.9, 0.99),
                'eps': 1e-8,
                'weight_decay': 0}
        if self.element_wise or self.simplified_model:
            self.use_siml_updater = False

        if self.num_workers is None:
            self.num_workers = util.determine_max_process()

        if (self.stop_trigger_epoch // self.log_trigger_epoch) == 0:
            raise ValueError(
                "Set stop_trigger_epoch larger than log_trigger_epoch")

        super().__post_init__()

    @property
    def output_skips(self):
        return self._collect_values(
            self.outputs, 'skip', default=False, asis=True)

    @property
    def input_names(self):
        return self._collect_values(
            self.inputs, 'name', asis=True)

    @property
    def input_dims(self):
        return self._collect_values(
            self.inputs, 'dim', default=1, asis=True)

    @property
    def output_names(self):
        return self._collect_values(
            self.outputs, 'name', asis=True)

    @property
    def output_dims(self):
        return self._collect_values(
            self.outputs, 'dim', default=1, asis=True)

    @property
    def input_length(self):
        return self._sum_dims(self.input_dims)

    @property
    def output_length(self):
        return self._sum_dims(self.output_dims)

    @property
    def variable_information(self):
        def to_dict(data):
            if isinstance(data, dict):
                return {v['name']: v for value in data.values() for v in value}
            elif isinstance(data, list):
                return {d['name']: d for d in data}
            else:
                raise ValueError(f"Unexpected data: {data}")
        out_dict = to_dict(self.inputs)
        out_dict.update(to_dict(self.outputs))
        return out_dict

    def update_output_directory(self, *, id_=None, base=None):
        if base is None:
            base = Path('models')
        else:
            base = Path(base)
        if id_ is None:
            self.output_directory = base \
                / f"{self.name}_{util.date_string()}"
        else:
            self.output_directory = base \
                / f"{self.name}_{id_}_{util.date_string()}"

    def _collect_values(self, data, key, *, default=None, asis=False):
        if default is None:
            def get(dict_data, key):
                return dict_data[key]
        else:
            def get(dict_data, key):
                return dict_data.get(key, default)

        if isinstance(data, list):
            return [get(d, key) for d in data]
        elif isinstance(data, dict):
            if asis:
                return {
                    dict_key: [get(v, key) for v in dict_value]
                    for dict_key, dict_value in data.items()}
            else:
                return [
                    get(v, key)
                    for dict_value in data.values() for v in dict_value]
        else:
            raise ValueError(f"Unexpected data: {data}")

    def _sum_dims(self, dim_data):
        if isinstance(dim_data, dict):
            return {key: np.sum(value) for key, value in dim_data.items()}
        else:
            return np.sum(dim_data)


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
    activation_after_residual: bool = True
    allow_linear_residual: bool = False
    bias: bool = True
    input_slice: slice = slice(0, None, 1)
    input_indices: list[int] = dc.field(
        default=None, metadata={'allow_none': True})
    input_keys: list[str] = dc.field(
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


@dc.dataclass
class ModelSetting(TypedDataClass):
    blocks: list[BlockSetting]

    def __init__(self, setting=None):
        if setting is None:
            self.blocks = [BlockSetting()]
        else:
            self.blocks = [
                BlockSetting(**block) for block in setting['blocks']]
        if np.all(b.is_first is False for b in self.blocks):
            self.blocks[0].is_first = True
        if np.all(b.is_last is False for b in self.blocks):
            self.blocks[-1].is_last = True


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
    """

    mandatory_variables: list[str] = dc.field(
        default_factory=list)
    optional_variables: list[str] = dc.field(
        default_factory=list)
    mandatory: list[str] = dc.field(
        default_factory=list)
    optional: list[str] = dc.field(
        default_factory=list)
    finished_file: str = 'converted'
    file_type: str = 'fistr'
    required_file_names: list[str] = dc.field(default_factory=list)
    skip_femio: bool = False
    time_series: bool = False
    save_femio: bool = False
    skip_save: bool = False

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
    data: DataSetting = DataSetting()
    conversion: ConversionSetting = ConversionSetting()
    preprocess: dict = dc.field(default_factory=dict)
    trainer: TrainerSetting = TrainerSetting()
    inferer: InfererSetting = InfererSetting()
    model: ModelSetting = ModelSetting()
    optuna: OptunaSetting = OptunaSetting()
    study: StudySetting = StudySetting()
    replace_preprocessed: bool = True

    @classmethod
    def read_settings_yaml(cls, settings_yaml, replace_preprocessed=True):
        dict_settings = util.load_yaml(settings_yaml)
        if isinstance(settings_yaml, Path):
            name = settings_yaml.stem
        else:
            name = None
        return cls.read_dict_settings(
            dict_settings, name=name,
            replace_preprocessed=replace_preprocessed)

    @classmethod
    def read_dict_settings(
            cls, dict_settings, *, name=None, replace_preprocessed=True):
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
            trainer_setting = TrainerSetting
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

        return cls(
            data=data_setting, conversion=conversion_setting,
            preprocess=preprocess_setting,
            trainer=trainer_setting, model=model_setting,
            inferer=inferer_setting,
            optuna=optuna_setting, study=study_setting,
            replace_preprocessed=replace_preprocessed)

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

    def update_with_dict(self, new_dict):
        original_dict = dc.asdict(self)
        return MainSetting.read_dict_settings(
            self._update_with_dict(original_dict, new_dict))

    def _update_with_dict(self, original_setting, new_setting):
        if isinstance(new_setting, str) or isinstance(new_setting, float) \
                or isinstance(new_setting, int):
            return new_setting
        elif isinstance(new_setting, list):
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


def write_yaml(data_class, file_name, *, overwrite=False):
    """Write YAML file of the specified dataclass object.

    Parameters
    -----------
    data_class: dataclasses.dataclass
        DataClass object to write.
    file_name: str or pathlib.Path
        YAML file name to write.
    overwrite: bool, optional
        If True, overwrite file.
    """
    file_name = Path(file_name)
    if file_name.exists() and not overwrite:
        raise ValueError(f"{file_name} already exists")

    with open(file_name, 'w') as f:
        dump_yaml(data_class, f)
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

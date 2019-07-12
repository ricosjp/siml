import dataclasses as dc
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
        """Convert all fields accordingly with their type definitions.

        Args:
            None
        Returns:
            None
        """
        for field_name, field in self.__dataclass_fields__.items():
            try:
                self._convert_field(field_name, field)
            except TypeError:
                raise TypeError(
                    f"Can't convert {getattr(self, field_name)} to "
                    f"{field.type}")

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

        if field.type == typing.List[Path]:
            def type_function(x):
                return [Path(_x) for _x in x]
        elif field.type == typing.List[str]:
            def type_function(x):
                return [str(_x) for _x in x]
        elif field.type == typing.List[int]:
            def type_function(x):
                return [int(_x) for _x in x]
        elif field.type == typing.List[float]:
            def type_function(x):
                return [float(_x) for _x in x]
        elif field.type == typing.List[dict]:
            def type_function(x):
                return [dict(_x) for _x in x]
        else:
            type_function = field.type

        setattr(
            self, field_name, type_function(getattr(self, field_name)))

    def _validate_field(self, field_name, field):
        return isinstance(getattr(self, field_name), field.type)

    def __post_init__(self):
        self.convert()
        # self.validate()


@dc.dataclass
class DataSetting(TypedDataClass):

    raw: Path = Path('data/raw')
    interim: Path = Path('data/interim')
    preprocessed: Path = Path('data/preprocessed')
    inferred: Path = Path('data/inferred')
    train: typing.List[Path] = dc.field(
        default_factory=lambda: [Path('data/preprocessed/train')])
    validation: typing.List[Path] = dc.field(
        default_factory=lambda: [Path('data/preprocessed/validation')])
    test: typing.List[Path] = dc.field(
        default_factory=lambda: [Path('data/preprocessed/test')])


@dc.dataclass
class DBSetting(TypedDataClass):
    servername: str
    username: str
    password: str
    use_sqlite: bool = False


@dc.dataclass
class TrainerSetting(TypedDataClass):

    """
    inputs: list of dict
        Variable names of inputs.
    outputs: list of dict
        Variable names of outputs.
    train_directories: list of str or pathlib.Path
        Training data directories.
    output_directory: str or pathlib.Path
        Output directory name.
    validation_directories: list of str or pathlib.Path, optional [[]]
        Validation data directories.
    restart_directory: str or pathlib.Path, optional [None]
        Directory name to be used for restarting.
    pretrain_directory: str or pathlib.Path, optional [None]
        Pretrained directory name.
    loss_function: chainer.FunctionNode,
            optional [chainer.functions.mean_squared_error]
        Loss function to be used for training.
    optimizer: chainer.Optimizer, optional [chainer.optimizers.Adam]
        Optimizer to be used for training.
    compute_accuracy: bool, optional [False]
        If True, compute accuracy.
    batch_size: int, optional [10]
        Batch size.
    n_epoch: int, optional [1000]
        The number of epochs.
    gpu_id: int, optional [-1]
        GPU ID. Specify non negative value to use GPU. -1 Meaning CPU.
    log_trigger_epoch: int, optional [1]
        The interval of logging of training. It is used for logging,
        plotting, and saving snapshots.
    stop_trigger_epoch: int, optional [10]
        The interval to check if training should be stopped. It is used
        for early stopping and pruning.
    optuna_trial: optuna.Trial, optional [None]
        Trial object used to perform optuna hyper parameter tuning.
    prune: bool, optional [False]
        If True and optuna_trial is given, prining would be performed.
    seed: str, optional [0]
        Random seed.
    """

    inputs: typing.List[dict] = dc.field(default_factory=list)
    support_input: str = dc.field(default=None, metadata={'allow_none': True})
    outputs: typing.List[dict] = dc.field(default_factory=list)

    input_names: typing.List[str] = dc.field(
        default=None, metadata={'allow_none': True})
    input_dims: typing.List[int] = dc.field(
        default=None, metadata={'allow_none': True})
    output_names: typing.List[str] = dc.field(
        default=None, metadata={'allow_none': True})
    output_dims: typing.List[int] = dc.field(
        default=None, metadata={'allow_none': True})
    output_directory: Path = None

    name: str = dc.field(
        default=None, metadata={'allow_none': True})
    batch_size: int = 1
    n_epoch: int = 100
    log_trigger_epoch: int = 1
    stop_trigger_epoch: int = 10

    validation_directories: typing.List[Path] = dc.field(
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
    optuna_trial: optuna.Trial = dc.field(
        default=None, metadata={'convert': False, 'allow_none': True})
    prune: bool = False
    snapshot_choise_method: str = 'best'
    seed: int = 0

    def __post_init__(self):
        self.input_names = [i['name'] for i in self.inputs]
        self.input_dims = [i['dim'] for i in self.inputs]
        self.output_names = [o['name'] for o in self.outputs]
        self.output_dims = [o['dim'] for o in self.outputs]
        if self.output_directory is None:
            self.update_output_directory()
        super().__post_init__()

    def update_output_directory(self, *, id_=None):
        if id_ is None:
            self.output_directory = Path('models') \
                / f"{self.name}_{util.date_string()}"
        else:
            self.output_directory = Path('models') \
                / f"{self.name}_{id_}_{util.date_string()}"


@dc.dataclass
class BlockSetting(TypedDataClass):
    name: str = 'mlp'
    nodes: typing.List[int] = dc.field(
        default_factory=lambda: [-1, -1])
    activations: typing.List[str] = dc.field(
        default_factory=lambda: ['identity'])
    dropouts: typing.List[float] = dc.field(default_factory=lambda: [0.])

    # Parameters for dynamic definition of layers
    hidden_nodes: int = dc.field(
        default=None, metadata={'allow_none': True})
    hidden_layers: int = dc.field(
        default=None, metadata={'allow_none': True})
    hidden_activation: str = 'rely'
    output_activation: str = 'identity'
    input_dropout: float = 0.0
    hidden_dropout: float = 0.5
    output_dropout: float = 0.0

    def __post_init__(self):

        # Dynamic definition of layers
        if self.hidden_nodes is not None and self.hidden_layers is not None:
            self.nodes = \
                [-1] + [self.hidden_nodes] * self.hidden_layers + [-1]
            self.activations = [self.hidden_activation] * self.hidden_layers \
                + [self.output_activation]
            self.dropouts = [self.input_dropout] \
                + [self.hidden_dropout] * (self.hidden_layers - 1) \
                + [self.output_dropout]
        if not(
                len(self.nodes) - 1 == len(self.activations)
                == len(self.dropouts)):
            raise ValueError('Block definition invalid')
        super().__post_init__()


@dc.dataclass
class ModelSetting(TypedDataClass):
    blocks: typing.List[BlockSetting]

    def __init__(self, setting=None):
        if setting is None:
            self.blocks = [BlockSetting()]
        else:
            self.blocks = [
                BlockSetting(**block) for block in setting['blocks']]


@dc.dataclass
class OptunaSetting(TypedDataClass):
    n_trial: int = 100
    hyperparameters: typing.List[dict] = dc.field(default_factory=list)
    setting: dict = dc.field(default_factory=dict)
    # trainer: dict = dc.field(default_factory=dict)
    # model: dict = dc.field(default_factory=dict)


@dc.dataclass
class MainSetting:
    data: DataSetting = DataSetting()
    trainer: TrainerSetting = TrainerSetting()
    model: ModelSetting = ModelSetting()
    optuna: OptunaSetting = OptunaSetting()

    @classmethod
    def read_settings_yaml(cls, settings_yaml):
        settings_yaml = Path(settings_yaml)

        dict_settings = util.load_yaml_file(settings_yaml)
        return cls.read_dict_settings(dict_settings, name=settings_yaml.stem)

    @classmethod
    def read_dict_settings(cls, dict_settings, *, name=None):
        if 'name' not in dict_settings['trainer']:
            dict_settings['trainer']['name'] = name

        if 'data' in dict_settings:
            data_setting = DataSetting(**dict_settings['data'])
        else:
            data_setting = DataSetting()
        if 'trainer' in dict_settings:
            trainer_setting = TrainerSetting(**dict_settings['trainer'])
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

        return cls(
            data=data_setting, trainer=trainer_setting, model=model_setting,
            optuna=optuna_setting)

    def __post_init__(self):
        input_length = np.sum(self.trainer.input_dims)
        output_length = np.sum(self.trainer.output_dims)

        # Infer input and output dimension if they are not specified.
        # NOTE: Basically Chainer can infer input dimension, but not the case
        # when chainer.functions.einsum is used.
        if self.model.blocks[0].nodes[0] < 0:
            self.model.blocks[0].nodes[0] = input_length
        if self.model.blocks[-1].nodes[-1] < 0:
            self.model.blocks[-1].nodes[-1] = output_length

    def update_with_dict(self, new_dict):
        original_dict = dc.asdict(self)
        return MainSetting.read_dict_settings(
            self._update_with_dict(original_dict, new_dict))

    def _update_with_dict(self, original_setting, new_setting):
        if isinstance(new_setting, str) or isinstance(new_setting, float) \
                or isinstance(new_setting, int):
            return new_setting
        elif isinstance(new_setting, list):
            # NOTE: Assume that data is complete under the list
            return new_setting
        elif isinstance(new_setting, dict):
            for key, value in new_setting.items():
                original_setting.update({
                    key: self._update_with_dict(original_setting[key], value)})
            return original_setting
        else:
            raise ValueError(f"Unknown data type: {new_setting.__class__}")


@dc.dataclass
class PreprocessSetting:
    data: DataSetting
    preprocess: dict

    @classmethod
    def read_settings_yaml(cls, settings_yaml):
        dict_settings = util.load_yaml_file(settings_yaml)
        data = DataSetting(**dict_settings['data'])
        preprocess = dict_settings['preprocess']
        return cls(data=data, preprocess=preprocess)


def write_yaml(data_class, file_name):
    """Write YAML file of the specified dataclass object.

    Args:
        file_name: str or pathlib.Path
            YAML file name to write.
    Returns:
        None
    """
    file_name = Path(file_name)
    if file_name.exists():
        raise ValueError(f"{file_name} already exists")

    dict_data = dc.asdict(data_class)
    standardized_dict_data = _standardize_data(dict_data)

    with open(file_name, 'w') as f:
        yaml.dump(standardized_dict_data, f)
    return


def _standardize_data(data):
    if isinstance(data, list):
        return [_standardize_data(d) for d in data]
    elif isinstance(data, dict):
        return {k: _standardize_data(v) for k, v in data.items()}
    elif isinstance(data, Path):
        return str(data)
    else:
        return data

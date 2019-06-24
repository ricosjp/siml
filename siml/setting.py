import dataclasses as dc
from pathlib import Path
import typing

import optuna
import yaml

from . import util


@dc.dataclass
class TypedDataClass:

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

    raw: Path
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
            self.output_directory = Path('models') \
                / f"{self.name}_{util.date_string()}"
        super().__post_init__()


@dc.dataclass
class BlockSetting(TypedDataClass):
    name: str
    nodes: typing.List[int]
    activations: typing.List[str]
    dropouts: typing.List[float]

    def __post_init__(self):
        if not(
            len(self.nodes) - 1 == len(self.activations) == len(self.dropouts)
        ):
            raise ValueError('Block definition invalid')
        super().__post_init__()


@dc.dataclass
class ModelSetting(TypedDataClass):
    blocks: typing.List[BlockSetting]

    def __init__(self, setting):
        self.blocks = [BlockSetting(**block) for block in setting['blocks']]


@dc.dataclass
class MainSetting:
    data: DataSetting
    trainer: TrainerSetting
    model: ModelSetting

    @classmethod
    def read_settings_yaml(cls, settings_yaml):
        settings_yaml = Path(settings_yaml)

        dict_settings = util.load_yaml_file(settings_yaml)
        data_setting = DataSetting(**dict_settings['data'])
        trainer_setting = TrainerSetting(**dict_settings['trainer'])
        model_setting = ModelSetting(dict_settings['model'])

        if trainer_setting.name is None:
            trainer_setting.name = settings_yaml.stem
        return cls(
            data=data_setting, trainer=trainer_setting, model=model_setting)


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

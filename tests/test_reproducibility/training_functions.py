from pathlib import Path
import yaml
import shutil
import torch
import argparse

from siml import setting
from siml import trainer


def get_output_directory(seed: int, n_index: int) -> Path:
    output_dir = Path(
        "./tests/test_reproducibility/"
        f"out/reproducibility_seed_{seed}_{n_index}"
    )
    return output_dir


def train_cpu_short(seed: int, output_directory: Path):

    init_seed = torch.initial_seed()

    yaml_file = Path('tests/data/linear/linear_short.yml')
    with open(yaml_file, 'r') as fr:
        yaml_content = yaml.load(fr, yaml.FullLoader)

    yaml_content['trainer']['seed'] = seed
    yaml_content['trainer']['output_directory'] \
        = output_directory

    main_setting = setting.MainSetting.read_dict_settings(
        yaml_content
    )

    tr = trainer.Trainer(main_setting)
    if tr.setting.trainer.output_directory.exists():
        shutil.rmtree(tr.setting.trainer.output_directory)

    _ = tr.train()

    torch.save(init_seed, f"{output_directory}/init_seed.pt")

    return init_seed


def _init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "seed",
        type=int,
        help="seed integer [int]"
    )
    parser.add_argument(
        "index",
        type=int,
        help='the index of tests'
    )
    return parser


if __name__ == "__main__":
    parser = _init_argparse()
    args = parser.parse_args()

    seed = args.seed
    n_index = args.index
    output_dir = get_output_directory(seed, n_index)

    train_cpu_short(seed, output_dir)

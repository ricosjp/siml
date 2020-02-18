import argparse
from distutils.util import strtobool
import pathlib

from .. import siml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'settings_yaml',
        type=pathlib.Path,
        help='YAML file name of settings.')
    parser.add_argument(
        '-g', '--gpu-id',
        type=int,
        default=-1,
        help='GPU ID [-1, meaning CPU]')
    parser.add_argument(
        '-v', '--plot-validation',
        type=strtobool,
        default=1,
        help='If True, plot also validation loss [True]')
    args = parser.parse_args()

    original_setting = siml.setting.MainSetting.read_settings_yaml(
        args.settings_yaml)
    original_setting.study.plot_validation = args.plot_validation

    study = siml.study.Study(original_setting)
    study.run()

    return


if __name__ == '__main__':
    main()

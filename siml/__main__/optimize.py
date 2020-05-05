import argparse
from distutils.util import strtobool
import pathlib

import siml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'settings_yaml',
        type=pathlib.Path,
        help='YAML file name of settings.')
    parser.add_argument(
        '-o', '--out-dir',
        type=pathlib.Path,
        default=None,
        help='Output directory name')
    parser.add_argument(
        '-g', '--gpu-id',
        type=int,
        default=-1,
        help='GPU ID [-1, meaning CPU]')
    parser.add_argument(
        '-d', '--db-settings-yaml',
        type=pathlib.Path,
        default=None,
        help='DB setting file [None, meaning the same as settings_yaml]')
    parser.add_argument(
        '-l', '--sqlite',
        type=strtobool,
        default=0,
        help='If True, use SQLite to save log.')
    parser.add_argument(
        '-s', '--step-by-step',
        type=strtobool,
        default=0,
        help='If True, stop after one optimization step finished [False]')
    args = parser.parse_args()

    main_setting = siml.setting.MainSetting.read_settings_yaml(
        args.settings_yaml)
    if args.sqlite:
        db_setting = siml.setting.DBSetting(use_sqlite=True)
    else:
        if args.db_settings_yaml is None:
            db_setting = None
        else:
            db_setting = siml.setting.DBSetting.read_settings_yaml(
                args.db_settings_yaml)

    if args.out_dir is not None:
        main_setting.trainer.out_dir(args.out_dir)
    main_setting.trainer.gpu_id = args.gpu_id

    study = siml.optimize.Study(
        main_setting, db_setting, step_by_step=args.step_by_step)
    study.perform_study()
    return


if __name__ == '__main__':
    main()

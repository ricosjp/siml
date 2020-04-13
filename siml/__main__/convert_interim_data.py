import argparse
from distutils.util import strtobool
import pathlib

import siml


def main():
    parser = argparse.ArgumentParser(
        'Convert interim data with deterimined preprocessor settings.')
    parser.add_argument(
        'settings_yaml',
        type=pathlib.Path,
        help='YAML file name of settings.')
    parser.add_argument(
        '-f', '--force-renew',
        type=strtobool,
        default=0,
        help='If True, overwrite existing data [False]')
    parser.add_argument(
        '-g', '--group-id',
        type=int,
        default=None,
        help='Group ID to preprocess variables partially [None]')
    args = parser.parse_args()

    preprocessor = siml.prepost.Preprocessor.read_settings(
        args.settings_yaml, force_renew=args.force_renew)
    preprocessor.merge_dict_preprocessor_setting_pkls()
    preprocessor.convert_interim_data(group_id=args.group_id)
    print('success')


if __name__ == '__main__':
    main()

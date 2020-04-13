import argparse
from distutils.util import strtobool
import pathlib

import siml


def main():
    parser = argparse.ArgumentParser(
        'Preprocess interim data.')
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
        default=0,
        help='Group ID to preprocess variables partially [0]')
    args = parser.parse_args()

    preprocessor = siml.prepost.Preprocessor.read_settings(
        args.settings_yaml, force_renew=args.force_renew)
    preprocessor.prepare_preprocess_converters(group_id=args.group_id)
    preprocessor.merge_dict_preprocessor_setting_pkls()
    preprocessor.convert_interim_data(group_id=args.group_id)

    print('success')
    return


if __name__ == '__main__':
    main()

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
    args = parser.parse_args()

    preprocessor = siml.prepost.Preprocessor.read_settings(
        args.settings_yaml, force_renew=args.force_renew)
    preprocessor.preprocess_interim_data()

    print('success')
    return


if __name__ == '__main__':
    main()

import argparse
from distutils.util import strtobool
import pathlib

from . import prepost


def convert_raw_data(
        add_argument=None, conversion_function=None, filter_function=None,
        load_function=None, **kwargs):
    parser = argparse.ArgumentParser()
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
        '-n', '--read-npy',
        type=strtobool,
        default=1,
        help='If True, read .npy files instead of original files '
        'if exists [True]')
    parser.add_argument(
        '-r', '--recursive',
        type=strtobool,
        default=1,
        help='If True, process directory recursively [True]')
    parser = add_argument(parser)
    args = parser.parse_args()

    raw_converter = prepost.RawConverter.read_settings(
        args.settings_yaml,
        conversion_function=conversion_function,
        filter_function=filter_function,
        load_function=load_function,
        force_renew=args.force_renew,
        recursive=args.recursive,
        read_npy=args.read_npy, **kwargs)
    raw_converter.convert()
    print('success')

    return

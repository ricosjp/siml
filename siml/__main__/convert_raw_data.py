import argparse
from distutils.util import strtobool
import pathlib

from siml.preprocessing import converter


def main(
        add_argument=None, conversion_function=None, filter_function=None,
        load_function=None):
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
    args = parser.parse_args()

    raw_converter = converter.RawConverter.read_settings(
        args.settings_yaml,
        conversion_function=conversion_function,
        filter_function=filter_function,
        load_function=load_function,
        force_renew=args.force_renew,
        recursive=args.recursive,
        to_first_order=True,
        write_ucd=False,
        read_npy=args.read_npy, read_res=False)
    results = raw_converter.convert()

    if results.query_num_status_items(status='failed') != 0:
        raise ValueError('Failed items are included. Check logs.')
    print('success')


if __name__ == '__main__':
    main()

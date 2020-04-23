import argparse
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
        '-r', '--restart-dir',
        type=pathlib.Path,
        default=None,
        help='Restart directory name')
    parser.add_argument(
        '-p', '--pretrained-directory',
        type=pathlib.Path,
        default=None,
        help='Pretrained directory name')
    args = parser.parse_args()

    main_setting = siml.setting.MainSetting.read_settings_yaml(
        args.settings_yaml)
    if args.out_dir is not None:
        main_setting.trainer.out_dir(args.out_dir)
    main_setting.trainer.gpu_id = args.gpu_id
    if args.restart_dir is not None:
        main_setting.trainer.restart_directory = args.restart_dir
    if args.pretrained_directory is not None:
        main_setting.trainer.pretrain_directory = args.pretrained_directory

    trainer = siml.trainer.Trainer(main_setting)
    trainer.train()


if __name__ == '__main__':
    main()

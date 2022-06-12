import argparse
from distutils.util import strtobool
import pathlib
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import siml


def main():
    parser = argparse.ArgumentParser(
        'Plot loss vs epoch curve')
    parser.add_argument(
        'data_directories',
        type=pathlib.Path,
        nargs='+',
        help='Data directory of interim or preprocessed data')
    parser.add_argument(
        '-o', '--out-dir',
        type=pathlib.Path,
        default=None,
        help='Output base directory name [None]')
    parser.add_argument(
        '-f', '--filter',
        type=str,
        default=None,
        help='Filter string to extract directories')
    parser.add_argument(
        '-v', '--inverse-filter',
        type=str,
        default=None,
        help='Filter string to exclude directories')
    parser.add_argument(
        '-k', '--sort-key',
        type=str,
        default='validation_loss',
        help='Key to sort log files [''validation_loss'']')
    parser.add_argument(
        '-t', '--plot-train-loss',
        type=strtobool,
        default=1,
        help='If True, plot train loss in addition to validation loss [True]')
    parser.add_argument(
        '-m', '--plot-minimum-loss',
        type=strtobool,
        default=1,
        help='If True, plot minimum loss point [True]')
    parser.add_argument(
        '-l', '--show-legend',
        type=strtobool,
        default=1,
        help='If True, show legend [True]')
    parser.add_argument(
        '-n', '--max-n-files',
        type=int,
        default=None,
        help='Maximum number of files')
    parser.add_argument(
        '-x', '--x-limit',
        type=float,
        nargs='+',
        default=None,
        help='X limit of the plot')
    parser.add_argument(
        '-y', '--y-limit',
        type=float,
        nargs='+',
        default=None,
        help='Y limit of the plot')
    parser.add_argument(
        '-a', '--x-axis',
        type=str,
        default='epoch',
        help='X axis name')
    parser.add_argument(
        '-g', '--log-x-axis',
        type=strtobool,
        default=0,
        help='If True, set x axis scale logarithmic')
    parser.add_argument(
        '-c', '--continuous',
        type=strtobool,
        default=0,
        help='If True, take into account assume continous training [True]')
    args = parser.parse_args()

    csv_files = siml.util.collect_files(
        args.data_directories, pattern=args.filter,
        inverse_pattern=args.inverse_filter, required_file_names=['log.csv'],
        allow_no_data=True)
    if len(csv_files) == 0:
        raise ValueError(f"No data found in: {args.data_directories}")
    valid_csv_files, data_frames = load_logs(csv_files)
    n_files = len(valid_csv_files)

    minimum_values = calculate_operated_values(
        data_frames, args.sort_key, op=np.min)
    maximum_values = calculate_operated_values(
        data_frames, args.sort_key, op=np.max)
    if args.max_n_files is not None:
        args.max_n_files = min(args.max_n_files, n_files)
    sorted_indices = np.argsort(minimum_values)[:args.max_n_files]
    n_seleced_files = len(sorted_indices)

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    cmap = mpl.cm.get_cmap('inferno')
    norm = mpl.colors.Normalize(
        vmin=0, vmax=n_seleced_files+3)  # Not to use too bright color
    plt.figure(figsize=(16, 9))
    styles = ['-', '--', '-.']

    label_flag = []
    for i, sorted_index in enumerate(sorted_indices):
        df = data_frames[sorted_index]
        name = valid_csv_files[sorted_index]
        if n_seleced_files == 1:
            alpha = 1.
            lw = 1.
        else:
            alpha = .5 + 0.5 * (
                n_seleced_files - 1 - i) / (n_seleced_files - 1)
            lw = 1. + 2. * (n_seleced_files - 1 - i) / (n_seleced_files - 1)
        if args.continuous:
            x_values, index = cumulate_x_values(
                name, df, valid_csv_files, data_frames, sorted_indices,
                x_axis=args.x_axis)
            color = cmap(norm(n_seleced_files - 1 - index))
        else:
            x_values = df[args.x_axis]
            color = cmap(norm(n_seleced_files - 1 - i))

        match = re.search(r'cont(\d+)', name)
        if match is None:
            focus_continue_step = 0
            root_string = str(pathlib.Path(name).parent)
        else:
            focus_continue_step = int(match.groups()[0])
            root_string = name.split(f"_cont{focus_continue_step}")[0]
        if root_string not in label_flag:
            plt.plot(
                x_values, df['validation_loss'], styles[i % len(styles)],
                color=color, alpha=alpha, lw=lw, label=name)
            label_flag.append(root_string)
        else:
            # Plot without label
            plt.plot(
                x_values, df['validation_loss'], styles[i % len(styles)],
                color=color, alpha=alpha, lw=lw)

        validation_min_index = np.argmin(df['validation_loss'].values)
        if args.plot_minimum_loss:
            plt.plot(
                x_values[validation_min_index],
                df['validation_loss'][validation_min_index],
                '*', ms=lw*5, color=color, alpha=alpha)
        print(f"--\nFile name: {name}")
        print(
            f"\t     Best epoch: {df['epoch'][validation_min_index]}\n"
            '\tValidation loss: '
            f"{df['validation_loss'][validation_min_index]}\n"
            f"\t     Train loss: {df['train_loss'][validation_min_index]}")
        if args.plot_train_loss:
            plt.plot(
                x_values, df['train_loss'], ':',
                color=color, alpha=alpha, lw=lw)

    plt.yscale('log')
    if args.log_x_axis:
        plt.xscale('log')
        plt.xlim(1, None)
    else:
        plt.xlim(0, None)
    plt.xlabel(args.x_axis)
    plt.ylabel('loss')
    if args.x_limit is not None:
        if len(args.x_limit) != 2:
            raise ValueError(
                f"Length of x_limit should be 2 ({len(args.x_limit)} given)")
        plt.xlim(args.x_limit)
    if args.y_limit is not None:
        if len(args.y_limit) != 2:
            raise ValueError(
                f"Length of y_limit should be 2 ({len(args.y_limit)} given)")
        plt.ylim(args.y_limit)
    if args.show_legend:
        plt.legend()

    if args.out_dir is None:
        plt.show()
    else:
        file_name = args.out_dir / f"losses_{siml.util.date_string()}.pdf"
        plt.savefig(file_name)
        print(f"Figure saved in: {file_name}")

    return


def cumulate_x_values(
        focus_file_name, focus_data_frame, file_names, data_frames,
        sorted_indices, x_axis):

    candidate_indices = [
        i for i, sorted_index in enumerate(sorted_indices)
        if focus_file_name == file_names[sorted_index]]
    if len(candidate_indices) != 1:
        raise ValueError(
            f"{len(candidate_indices)} files found for: {focus_file_name}")
    index = candidate_indices[0]

    match = re.search(r'cont(\d+)', focus_file_name)
    if match is None:
        focus_continue_step = 0
        root_string = str(pathlib.Path(focus_file_name).parent)
    else:
        focus_continue_step = int(match.groups()[0])
        root_string = focus_file_name.split(f"_cont{focus_continue_step}")[0]

    x_offset = 0
    for data_frame, file_name in zip(data_frames, file_names):
        if root_string not in file_name:
            continue
        match = re.search(r'cont(\d+)', file_name)
        if match is None:
            continue_step = 0
        else:
            continue_step = int(match.groups()[0])

        if continue_step < focus_continue_step:
            x_offset += np.max(data_frame[x_axis].values)

        candidate_indices = [
            i for i, sorted_index in enumerate(sorted_indices)
            if file_name == file_names[sorted_index]]
        if len(candidate_indices) == 1:
            index = min(index, candidate_indices[0])

    return x_offset + focus_data_frame[x_axis].values, index


def load_logs(csv_files):
    data_frames = [
        pd.read_csv(f, header=0, index_col=None, skipinitialspace=True)
        for f in csv_files]
    valid_csv_files = [
        csv_files[i] for i, d in enumerate(data_frames) if len(d) > 0]
    valid_data_frames = [d for d in data_frames if len(d) > 0]

    return valid_csv_files, valid_data_frames


def calculate_operated_values(data_frames, sort_key, op=np.min):
    return [op(df[sort_key].values) for df in data_frames]


if __name__ == '__main__':
    main()

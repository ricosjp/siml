import argparse
import pathlib
from distutils.util import strtobool

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
    args = parser.parse_args()

    csv_files = siml.util.collect_files(
        args.data_directories, pattern=args.filter,
        inverse_pattern=args.inverse_filter, required_file_names=['log.csv'])
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

    for i, sorted_index in enumerate(sorted_indices):
        df = data_frames[sorted_index]
        name = valid_csv_files[sorted_index]
        alpha = .5 + 0.5 * (n_seleced_files - 1 - i) / (n_seleced_files - 1)
        lw = 1. + 2. * (n_seleced_files - 1 - i) / (n_seleced_files - 1)
        plt.plot(
            df['epoch'], df['validation_loss'], styles[i % len(styles)],
            color=cmap(norm(n_seleced_files - 1 - i)),
            alpha=alpha, lw=lw, label=name)
        validation_min_index = np.argmin(df['validation_loss'].values)
        if args.plot_minimum_loss:
            plt.plot(
                df['epoch'][validation_min_index],
                df['validation_loss'][validation_min_index],
                '*', ms=lw*5, color=cmap(norm(n_seleced_files - 1 - i)),
                alpha=alpha)
        print(f"--\nFile name: {name}")
        print(
            f"\t     Best epoch: {df['epoch'][validation_min_index]}\n"
            '\tValidation loss: '
            f"{df['validation_loss'][validation_min_index]}\n"
            f"\t     Train loss: {df['train_loss'][validation_min_index]}")
        if args.plot_train_loss:
            plt.plot(
                df['epoch'], df['train_loss'], ':',
                color=cmap(norm(n_seleced_files - 1 - i)),
                alpha=alpha, lw=lw)

    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim(0, None)
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

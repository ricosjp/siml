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
        '-n', '--max-n-files',
        type=int,
        default=None,
        help='Maximum number of files')
    args = parser.parse_args()

    csv_files = siml.util.collect_files(
        args.data_directories, pattern=args.filter,
        required_file_names=['log.csv'])
    n_files = len(csv_files)

    data_frames = load_logs(csv_files)
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
    for i, sorted_index in enumerate(sorted_indices[::-1]):
        df = data_frames[sorted_index]
        alpha = .5 + 0.5 * i / (n_seleced_files - 1)
        lw = 1. + 2. * i / (n_seleced_files - 1)
        plt.plot(
            df['epoch'], df['validation_loss'], styles[i % len(styles)],
            color=cmap(norm(i)), alpha=alpha, lw=lw)
        if args.plot_train_loss:
            plt.plot(
                df['epoch'], df['train_loss'], ':',
                color=cmap(norm(i)), alpha=alpha, lw=lw)

    # Dummpy plot for legends
    for i, sorted_index in enumerate(sorted_indices):
        df = data_frames[sorted_index]
        name = csv_files[sorted_index]
        plt.plot(
            [], [], styles[(n_seleced_files - 1 - i) % len(styles)],
            color=cmap(norm(n_seleced_files - 1 - i)), label=name)

    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim(0, None)
    plt.ylim(None, 10**np.ceil(np.log10(np.max(maximum_values))))
    plt.legend()

    if args.out_dir is None:
        plt.show()
    else:
        file_name = args.out_dir / f"losses_{siml.util.date_string()}.pdf"
        plt.savefig(file_name)
        print(f"Figure saved in: {file_name}")

    return


def load_logs(csv_files):
    return [
        pd.read_csv(f, header=0, index_col=None, skipinitialspace=True)
        for f in csv_files]


def calculate_operated_values(data_frames, sort_key, op=np.min):
    return [op(df[sort_key].values) for df in data_frames]


if __name__ == '__main__':
    main()

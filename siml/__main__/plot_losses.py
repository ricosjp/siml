import argparse
import glob
import pathlib

import femio
import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplot3d
import scipy.sparse as sp


def main():
    parser = argparse.ArgumentParser(
        'Visualize graph data')
    parser.add_argument(
        'data_directory',
        type=pathlib.Path,
        help='Data directory of interim or preprocessed data')
    parser.add_argument(
        '-r', '--range',
        type=float,
        default=None,
        help='Range of color of edge weights')
    args = parser.parse_args()

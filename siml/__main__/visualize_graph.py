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
import siml


def main():
    parser = argparse.ArgumentParser(
        'Visualize graph data')
    parser.add_argument(
        'data_directory',
        type=pathlib.Path,
        help='Data directory of interim or preprocessed data')
    parser.add_argument(
        '-o', '--out-dir',
        type=pathlib.Path,
        default=None,
        help='Output base directory name [None]')
    parser.add_argument(
        '-r', '--range',
        type=float,
        default=None,
        help='Range of color of edge weights')
    args = parser.parse_args()

    npz_files = glob.glob(str(args.data_directory / '*.npz'))
    nodes = load_nodes(args.data_directory)

    if args.out_dir is not None:
        output_directory = args.out_dir / args.data_directory.name \
            / siml.util.date_string()
        output_directory.mkdir(parents=True, exist_ok=True)
    for npz_file in npz_files:
        try:
            sparse_matrix = sp.load_npz(npz_file)
        except ValueError:
            print(f"{npz_file} is not sparse matrix data.")
            continue

        graph = nx.from_scipy_sparse_matrix(
            sparse_matrix, parallel_edges=False, create_using=nx.DiGraph)
        for i, node in enumerate(nodes):
            graph.add_node(i, pos=node)
        name = pathlib.Path(npz_file).stem
        plot_network_3d(
            graph, nodes, name=name, range_=args.range)

        if args.out_dir is not None:
            file_name = output_directory / f"graph_{name}.pdf"
            plt.savefig(file_name)
            print(f"Figure saved in: {file_name}")

    if args.out_dir is None:
        plt.show()
    return


def load_nodes(data_directory):
    if (data_directory / 'femio_nodal_data.npz').is_file():
        fem_data = femio.FEMData.read_npy_directory(
            data_directory, read_mesh_only=True)
        nodes = fem_data.nodes.data
    elif (data_directory / 'nodes.npy').is_file():
        nodes = np.load(data_directory / 'nodes.npy')
    elif (data_directory / 'node.npy').is_file():
        nodes = np.load(data_directory / 'node.npy')
    else:
        nodes = None

    return nodes


def plot_network_3d(graph, positions, name, *, range_=None):

    with plt.style.context(('ggplot')):
        fig = plt.figure(figsize=(10, 7))
        ax = mplot3d.Axes3D(fig)

        for position in positions:
            xi = position[0]
            yi = position[1]
            zi = position[2]

            ax.scatter(xi, yi, zi, color='k', alpha=0.7)

        if range_ is None:
            weights = np.array(list(
                nx.get_edge_attributes(graph, 'weight').values()))
            range_ = np.max(np.abs(weights))
            print(f"abs_max: {range_}")
        cmap = mpl.cm.get_cmap('jet')
        norm = mpl.colors.Normalize(
            vmin=-range_, vmax=range_)  # pylint: disable=E1130
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = plt.colorbar(mappable, shrink=.3)
        colorbar.set_label('weight')
        colors = [
            cmap(norm(graph[start][end]['weight']))
            for start, end in graph.edges()]
        segs = np.array([
            np.stack([
                positions[start], (positions[start] + positions[end]) / 2])
            for start, end in graph.edges()])
        line_segments = mplot3d.art3d.Line3DCollection(
            segs, colors=colors, alpha=.5)
        ax.add_collection(line_segments)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(name)

    return fig


def scale_color(value, abs_max, min_=0, max_=255):
    value_0_to_1 = (value + abs_max) / abs_max / 2
    return value_0_to_1 * (max_ - min_) + min_


if __name__ == '__main__':
    main()

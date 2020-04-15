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
    args = parser.parse_args()

    npz_files = glob.glob(str(args.data_directory / '*.npz'))
    nodes = load_nodes(args.data_directory)
    for npz_file in npz_files:
        try:
            sparse_matrix = sp.load_npz(npz_file)
        except ValueError:
            print(f"{npz_file} is not sparse matrix data.")
            continue

        graph = nx.from_scipy_sparse_matrix(sparse_matrix)
        for i, node in enumerate(nodes):
            graph.add_node(i, pos=node)
        plot_network_3d(graph, nodes, name=pathlib.Path(npz_file).stem)

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


def plot_network_3d(graph, positions, name):

    with plt.style.context(('ggplot')):
        fig = plt.figure(figsize=(10, 7))
        ax = mplot3d.Axes3D(fig)

        for position in positions:
            xi = position[0]
            yi = position[1]
            zi = position[2]

            ax.scatter(xi, yi, zi, color='k', alpha=0.7)

        weights = np.array(list(
            nx.get_edge_attributes(graph, 'weight').values()))
        abs_max = np.max(np.abs(weights)) * .7
        cmap = mpl.cm.get_cmap('jet')
        norm = mpl.colors.Normalize(vmin=-abs_max, vmax=abs_max)
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = plt.colorbar(mappable, shrink=.3)
        colorbar.set_label('weight')
        print(f"abs_max: {abs_max}")
        colors = [
            cmap(norm(graph[start][end]['weight']))
            for start, end in graph.edges()]
        segs = np.array([
            np.stack([positions[start], positions[end]])
            for start, end in graph.edges()])
        line_segments = mplot3d.art3d.Line3DCollection(
            segs, colors=colors, alpha=.5)
        ax.add_collection(line_segments)

        # for start, end in graph.edges():
        #     # pos_start = positions[start]
        #     # pos_end = positions[end]
        #     # line_segments = mplot3d.art3d.Line3DCollection(
        #     #     np.stack([pos_start, pos_end])[None, :],
        #     #     colors=(colors(norm(graph[start][end]['weight'])),))
        #     # ax.add_collection(line_segments)
        #     x = np.array((positions[start][0], positions[end][0]))
        #     y = np.array((positions[start][1], positions[end][1]))
        #     z = np.array((positions[start][2], positions[end][2]))
        #     ax.plot(
        #         x, y, z,
        #         c=colors(norm(graph[start][end]['weight'])), alpha=1.)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(name)

    plt.show()
    return


def scale_color(value, abs_max, min_=0, max_=255):
    value_0_to_1 = (value + abs_max) / abs_max / 2
    return value_0_to_1 * (max_ - min_) + min_


if __name__ == '__main__':
    main()

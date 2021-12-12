
import femio
import numpy as np
import scipy.sparse as sp


EPS = 1.e-5


def main():
    for _ in range(10):
        generate_data(_)
    return


def generate_data(data_number):

    # Geometry
    fem_data_1 = femio.generate_brick(
        'tet', 2, 2, 2, x_length=1., y_length=1., z_length=1.)
    fem_data_2 = femio.generate_brick(
        'tet', 4, 4, 4, x_length=2., y_length=2., z_length=2.)
    fem_data_2.nodes.data[:, 0] = fem_data_2.nodes.data[:, 0] + 1.

    # Laplacian
    gx_1, gy_1, gz_1 = \
        fem_data_1.calculate_spatial_gradient_adjacency_matrices(
            'nodal', n_hop=1, moment_matrix=True,
            normals=True,
            consider_volume=False, normal_weight_factor=1.)
    lap_1 = gx_1.dot(gx_1) + gy_1.dot(gy_1) + gz_1.dot(gz_1)
    gx_2, gy_2, gz_2 = \
        fem_data_2.calculate_spatial_gradient_adjacency_matrices(
            'nodal', n_hop=1, moment_matrix=True,
            normals=True,
            consider_volume=False, normal_weight_factor=1.)
    lap_2 = gx_2.dot(gx_2) + gy_2.dot(gy_2) + gz_2.dot(gz_2)
    minv_1 = fem_data_1.nodal_data.get_attribute_data(
        'inversed_moment_tensors')[..., None]
    wnorm_1 = fem_data_1.nodal_data.get_attribute_data(
        'weighted_surface_normals')[..., None]
    minv_2 = fem_data_2.nodal_data.get_attribute_data(
        'inversed_moment_tensors')[..., None]
    wnorm_2 = fem_data_2.nodal_data.get_attribute_data(
        'weighted_surface_normals')[..., None]

    # Incidence matrix
    incidence_2to1 = sp.coo_matrix(
        (
            np.array([1] * 9),
            (
                np.array([3, 6, 9, 12, 15, 18, 21, 24, 27]) - 1,
                np.array([1, 6, 11, 26, 31, 36, 51, 56, 61]) - 1,
            )
        ), shape=(len(fem_data_1.nodes), len(fem_data_2.nodes)))
    sp.save_npz('incidence_2to1.npz', incidence_2to1)

    # Periodic BC
    y_max_indices = np.where(np.abs(
        fem_data_2.nodes.data[:, 1] - np.max(fem_data_2.nodes.data[:, 1])
    ) < EPS)[0]
    y_min_indices = np.where(np.abs(
        fem_data_2.nodes.data[:, 1] - np.min(fem_data_2.nodes.data[:, 1])
    ) < EPS)[0]
    periodic_2 = sp.coo_matrix(
        (
            np.array([1] * len(y_max_indices) * 2),
            (
                np.concatenate([y_max_indices, y_min_indices]),
                np.concatenate([y_min_indices, y_max_indices]),
            ),
        ), shape=(len(fem_data_2.nodes), len(fem_data_2.nodes)))
    sp.save_npz('periodic_2.npz', periodic_2)
    np.testing.assert_almost_equal(
        (periodic_2.T - periodic_2).sum(axis=1), 0)

    # Initial condition
    fem_data_1.nodal_data.update_data(
        fem_data_1.nodes.ids, {
            'phi_0':
            np.sin((
                fem_data_1.nodes.data[:, [0]] * np.random.rand() / 4
                + np.random.rand()) * np.pi * 2)})
    fem_data_2.nodal_data.update_data(
        fem_data_2.nodes.ids, {
            'phi_0':
            np.sin((
                fem_data_2.nodes.data[:, [0]] * np.random.rand() / 4
                + np.random.rand()) * np.pi * 2)})

    # Time evolution
    phi_1_1 = np.copy(fem_data_1.nodal_data.get_attribute_data('phi_0'))
    phi_2_1 = np.copy(fem_data_2.nodal_data.get_attribute_data('phi_0'))

    row = periodic_2.row
    phi_2_1[row] = (phi_2_1[row] + periodic_2.dot(phi_2_1)[row]) / 2

    coeff = 1e-5 * (1 + np.random.rand())
    heat_transfer = 1. * (1 + np.random.rand())

    n_substep = 1000
    for i_repeat in range(20 * n_substep + 1):
        heat_2to1 = np.zeros((len(fem_data_1.nodes), 3, 1))
        heat_2to1[incidence_2to1.row] = heat_transfer * np.einsum(
            'ilkf,ikf->ilf',
            minv_1[incidence_2to1.row],
            np.einsum(
                'if,ikf->ikf',
                incidence_2to1.dot(phi_2_1)[incidence_2to1.row]
                - phi_1_1[incidence_2to1.row],
                wnorm_1[incidence_2to1.row]))

        heat_1to2 = np.zeros((len(fem_data_2.nodes), 3, 1))
        heat_1to2[incidence_2to1.T.row] = heat_transfer * np.einsum(
            'ilkf,ikf->ilf',
            minv_2[incidence_2to1.T.row],
            np.einsum(
                'if,ikf->ikf',
                incidence_2to1.T.dot(phi_1_1)[incidence_2to1.T.row]
                - phi_2_1[incidence_2to1.T.row],
                wnorm_2[incidence_2to1.T.row]))
        print(np.sum(heat_2to1), np.sum(heat_1to2))
        dphi_1 = lap_1.dot(phi_1_1) + (
            + gx_1.dot(heat_2to1[:, 0])
            + gy_1.dot(heat_2to1[:, 1])
            + gz_1.dot(heat_2to1[:, 2]))
        dphi_2 = lap_2.dot(phi_2_1) + (
            + gx_2.dot(heat_1to2[:, 0])
            + gy_2.dot(heat_1to2[:, 1])
            + gz_2.dot(heat_1to2[:, 2]))
        phi_1_1 += coeff * dphi_1
        phi_2_1 += coeff * dphi_2

        row = periodic_2.row
        phi_2_1[row] = (phi_2_1[row] + periodic_2.dot(phi_2_1)[row]) / 2

        # if i_repeat % 1000 == 0:
        #     fem_data_1.nodal_data.update_data(
        #         fem_data_1.nodes.ids, {
        #             'phi': phi_1_1,
        #             'heat': heat_2to1[..., 0],
        #         }, allow_overwrite=True)
        #     fem_data_2.nodal_data.update_data(
        #         fem_data_2.nodes.ids, {
        #             'phi': phi_2_1,
        #             'heat': heat_1to2[..., 0],
        #         }, allow_overwrite=True)
        #     fem_data_1.write(
        #         'ucd', f"series/raw/1.{int(i_repeat / n_substep)}.inp",
        #         overwrite=True)
        #     fem_data_2.write(
        #         'ucd', f"series/raw/2.{int(i_repeat / n_substep)}.inp",
        #         overwrite=True)

    fem_data_1.nodal_data.update_data(
        fem_data_1.nodes.ids, {'phi_1': phi_1_1})
    fem_data_2.nodal_data.update_data(
        fem_data_2.nodes.ids, {'phi_1': phi_2_1})

    # Write
    fem_data_1.write('ucd', f"raw/{data_number}/mesh_1.inp", overwrite=True)
    fem_data_2.write('ucd', f"raw/{data_number}/mesh_2.inp", overwrite=True)
    np.save(f"raw/{data_number}/coeff.npy", coeff)
    np.save(f"raw/{data_number}/heat_transfer.npy", heat_transfer)
    return


if __name__ == '__main__':
    main()

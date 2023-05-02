import pathlib
import shutil

import numpy as np

from siml.path_like_objects import SimlFileBulider
from siml.preprocessing.siml_scalers import SimlScalerWrapper
from siml.preprocessing.siml_scalers.scale_functions import StandardScaler


def test__std_scale_values():
    n_element = int(1e5)
    epsilon = 1e-5

    interim_std_scale = np.random.rand(n_element, 3)

    scaler = StandardScaler(with_mean=False)
    preprocessed_data = scaler.fit_transform(interim_std_scale)

    np.testing.assert_almost_equal(
        interim_std_scale / (np.std(interim_std_scale, axis=0) + epsilon),
        preprocessed_data,
        decimal=3
    )

    # After transformed, std is almost 1
    np.testing.assert_almost_equal(
        np.std(preprocessed_data),
        1. + epsilon,
        decimal=3
    )


def test__standardize_values():
    n_element = int(1e5)
    epsilon = 1e-5

    interim_value = np.random.randn(n_element, 5) * 2 \
        + np.array([[.1, .2, .3, .4, .5]])

    scaler = StandardScaler()
    preprocessed = scaler.fit_transform(interim_value)

    np.testing.assert_almost_equal(
        (interim_value - np.mean(interim_value, axis=0))
        / (np.std(interim_value, axis=0) + epsilon),
        preprocessed,
        decimal=3
    )

    # After transformed, std is almost 1 and mean is almost zero
    np.testing.assert_almost_equal(
        np.std(preprocessed, axis=0),
        1. + epsilon,
        decimal=3
    )
    np.testing.assert_almost_equal(
        np.mean(preprocessed, axis=0),
        np.zeros(5),
        decimal=3
    )


def test__standardizer():
    n_data = 5
    dim = 3
    list_data = [
        np.random.randn(np.random.randint(2, 1e4), dim) * 2. * i + .5 * i
        for i in range(n_data)
    ]
    out_directory = pathlib.Path('tests/data/util_std')
    shutil.rmtree(out_directory, ignore_errors=True)
    data_files = [out_directory / f"data_{i}/x.npy" for i in range(n_data)]
    for data_file, d in zip(data_files, list_data):
        data_file.parent.mkdir(parents=True)
        np.save(data_file, d)

    once_std = SimlScalerWrapper('standardize')
    all_data = np.concatenate(list_data)
    once_std.partial_fit(all_data)

    data_files = [
        SimlFileBulider.create(f) for f in data_files
    ]
    lazy_std = SimlScalerWrapper('standardize')
    lazy_std.lazy_partial_fit(data_files)

    np.testing.assert_almost_equal(
        once_std.converter.mean_, lazy_std.converter.mean_)
    np.testing.assert_almost_equal(
        once_std.converter.var_, lazy_std.converter.var_)

    new_data = np.random.rand(100, dim)
    transformed_new_data = \
        (new_data - np.mean(all_data, axis=0)) / np.std(all_data, axis=0)

    np.testing.assert_almost_equal(
        lazy_std.transform(new_data),
        transformed_new_data
    )
    np.testing.assert_almost_equal(
        lazy_std.inverse_transform(transformed_new_data),
        new_data
    )


def test__standardizer_with_nan():
    n_data = 5
    dim = 3
    all_data = np.random.rand(n_data, 10, dim)
    all_data[0] = np.nan
    all_data[2, :, 0] = np.nan
    all_data[3, -1, 2] = np.nan

    out_directory = pathlib.Path('tests/data/util_std_with_nan')
    shutil.rmtree(out_directory, ignore_errors=True)
    data_files = [out_directory / f"data_{i}/x.npy" for i in range(n_data)]
    for data_file, d in zip(data_files, all_data):
        data_file.parent.mkdir(parents=True)
        np.save(data_file, d)
    data_files = [
        SimlFileBulider.create(f) for f in data_files
    ]

    lazy_std = SimlScalerWrapper('standardize', componentwise=False)
    lazy_std.lazy_partial_fit(data_files)
    assert not np.any(np.isnan(lazy_std.converter.mean_))
    assert not np.any(np.isnan(lazy_std.converter.var_))


def test__standardizer_with_nan_componentwise():
    n_data = 5
    dim = 3
    all_data = np.random.rand(n_data, 10, dim)
    all_data[0] = np.nan
    all_data[2, :, 0] = np.nan
    all_data[3, -1, 2] = np.nan

    out_directory = pathlib.Path('tests/data/util_std_with_nan')
    shutil.rmtree(out_directory, ignore_errors=True)
    data_files = [out_directory / f"data_{i}/x.npy" for i in range(n_data)]
    for data_file, d in zip(data_files, all_data):
        data_file.parent.mkdir(parents=True)
        np.save(data_file, d)

    data_files = [
        SimlFileBulider.create(f) for f in data_files
    ]
    lazy_std = SimlScalerWrapper(
        'standardize', componentwise=True
    )
    lazy_std.lazy_partial_fit(data_files)

    assert not np.any(np.isnan(lazy_std.converter.mean_))
    assert not np.any(np.isnan(lazy_std.converter.var_))


def test__std_scale():
    n_data = 5
    dim = 3
    list_data = [
        np.random.randn(np.random.randint(2, 1e4), dim) * 2. * i + .5 * i
        for i in range(n_data)]
    out_directory = pathlib.Path('tests/data/util_std')
    shutil.rmtree(out_directory, ignore_errors=True)
    data_files = [out_directory / f"data_{i}/x.npy" for i in range(n_data)]
    for data_file, d in zip(data_files, list_data):
        data_file.parent.mkdir(parents=True)
        np.save(data_file, d)

    all_data = np.concatenate(list_data)
    once_std = SimlScalerWrapper('std_scale')
    once_std.partial_fit(all_data)

    data_files = [
        SimlFileBulider.create(f) for f in data_files
    ]
    lazy_std = SimlScalerWrapper('std_scale')
    lazy_std.lazy_partial_fit(data_files)

    np.testing.assert_almost_equal(
        once_std.converter.var_, lazy_std.converter.var_)

    new_data = np.random.rand(100, dim)
    transformed_new_data = new_data / np.std(all_data, axis=0)
    np.testing.assert_almost_equal(
        lazy_std.transform(new_data), transformed_new_data)
    np.testing.assert_almost_equal(
        lazy_std.inverse_transform(transformed_new_data), new_data)

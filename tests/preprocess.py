import pathlib
import shutil

import numpy as np
import siml.prepost as prepost
import siml.setting as setting


def conversion_function(fem_data, data_directory):
    adj, _ = fem_data.calculate_adjacency_matrix_element()
    nadj = prepost.normalize_adjacency_matrix(adj)
    x_grad, y_grad, z_grad = \
        fem_data.calculate_spatial_gradient_adjacency_matrices('elemental')
    global_modulus = np.mean(
        fem_data.access_attribute('modulus'), keepdims=True)
    return {
        'adj': adj, 'nadj': nadj, 'global_modulus': global_modulus,
        'x_grad': x_grad, 'y_grad': y_grad, 'z_grad': z_grad}


def preprocess_deform():
    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/deform/data.yml'))

    raw_converter = prepost.RawConverter(
        main_setting, recursive=True, force_renew=True,
        conversion_function=conversion_function)
    raw_converter.convert()

    preprocessor = prepost.Preprocessor(main_setting, force_renew=True)
    preprocessor.preprocess_interim_data()


def preprocess_deform_timeseries():
    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/deform_timeseries/data.yml'))

    preprocessor = prepost.Preprocessor(main_setting, force_renew=True)
    preprocessor.preprocess_interim_data()


def preprocess_linear():
    p = prepost.Preprocessor.read_settings(
        pathlib.Path('tests/data/linear/linear.yml'), force_renew=True)
    p.preprocess_interim_data()


def generate_large():
    n_feat = 10
    n_element = 2000

    def generate_data(root_dir, n_data):
        if root_dir.exists():
            shutil.rmtree(root_dir)
        for i in range(n_data):
            r1 = np.random.rand()
            r2 = np.random.rand()
            floor = min(r1, r2)
            ceil = max(r1, r2)

            x = np.random.rand(n_element, n_feat) * (ceil - floor) + floor
            y = np.sin(x * 4. * np.pi)

            output_directory = root_dir / f"{i}"
            output_directory.mkdir(parents=True)
            np.save(output_directory / 'x.npy', x.astype(np.float32))
            np.save(output_directory / 'y.npy', y.astype(np.float32))

    output_root = pathlib.Path('tests/data/large/preprocessed')
    train_root = output_root / 'train'
    n_train_data = 20
    generate_data(train_root, n_train_data)

    validation_root = output_root / 'validation'
    n_validation_data = 2
    generate_data(validation_root, n_validation_data)

    test_root = output_root / 'test'
    n_test_data = 2
    generate_data(test_root, n_test_data)
    return


if __name__ == '__main__':
    preprocess_deform()
    preprocess_deform_timeseries()
    preprocess_linear()
    generate_large()

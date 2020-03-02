import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import siml.prepost as prepost
import siml.setting as setting


PLOT = False


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


def generate_ode():
    time_range = (10., 50.)
    delta_t = .1

    def f0(ts, xs):
        ys = np.zeros(list(xs.shape[:2]) + [1])
        ys[0] = np.random.rand(*ys.shape[1:])
        for i in range(1, len(ts)):
            ys[i, :, 0] = ys[i - 1, :, 0] + delta_t * (- .1 * ys[i - 1, :, 0])
        return ys

    def f1(ts, xs):
        ys = np.zeros(list(xs.shape[:2]) + [1])
        ys[0] = np.random.rand(*ys.shape[1:])
        for i in range(1, len(ts)):
            ys[i, :, 0] = ys[i - 1, :, 0] + delta_t * (
                .3 * xs[i, :, 0] + .2 * xs[i, :, 1] * xs[i, :, 2]
                - ys[i - 1, :, 0])
        return ys

    def f2(ts, xs):
        ys = np.zeros(list(xs.shape[:2]) + [1])
        ys[0] = np.random.rand(*ys.shape[1:])
        for i in range(1, len(ts)):
            ys[i, :, 0] = ys[i - 1, :, 0] + delta_t * (
                .1 * xs[i, :, 1] + .5 * xs[i, :, 0] * xs[i, :, 3]
                + (1 - ys[i - 1, :, 0]**2) * ys[i - 1, :, 0])
        return ys

    def generate_ode(root_dir, n_data):
        if root_dir.exists():
            shutil.rmtree(root_dir)

        for i in range(n_data):
            n_element = np.random.randint(3, 10)
            t_max = np.random.rand() * (
                time_range[1] - time_range[0]) + time_range[0]
            ts = np.arange(0., t_max, delta_t)
            x0 = np.random.rand() * np.sin(
                2 * np.pi * (np.random.rand() / 10. * ts + np.random.rand()))
            x1 = np.random.rand() * np.sin(
                2 * np.pi * (np.random.rand() / 20. * ts + np.random.rand()))
            x2 = np.random.rand() * (
                1 - np.exp(- ts / 5. * np.random.rand())) + np.random.rand()
            x3 = np.exp(- ts / 10. * np.random.rand()) + np.random.rand()
            _xs = np.stack([x0, x1, x2, x3], axis=1)[:, None, :]
            xs = np.concatenate([
                _xs * a for a in np.linspace(1., 2., n_element)], axis=1)

            y0 = f0(ts, xs)
            y1 = f1(ts, xs)
            y2 = f2(ts, xs)

            stacked_ts = np.concatenate(
                [ts[:, None, None]] * n_element, axis=1)

            output_directory = root_dir / f"{i}"
            output_directory.mkdir(parents=True)
            np.save(output_directory / 't.npy', stacked_ts.astype(np.float32))
            np.save(output_directory / 'x.npy', xs.astype(np.float32))
            np.save(output_directory / 'y0.npy', y0.astype(np.float32))
            np.save(output_directory / 'y1.npy', y1.astype(np.float32))
            np.save(output_directory / 'y2.npy', y2.astype(np.float32))
            np.save(
                output_directory / 'y0_initial.npy',
                (np.ones(y0.shape) * y0[0, :, :]).astype(np.float32))
            np.save(
                output_directory / 'y1_initial.npy',
                (np.ones(y0.shape) * y1[0, :, :]).astype(np.float32))
            np.save(
                output_directory / 'y2_initial.npy',
                (np.ones(y0.shape) * y2[0, :, :]).astype(np.float32))
            (output_directory / 'converted').touch()

            if PLOT:
                plt.plot(ts, x0, label='x0')
                plt.plot(ts, x1, label='x1')
                plt.plot(ts, x2, label='x2')
                plt.plot(ts, x3, label='x3')
                plt.plot(ts, y0[:, 0, 0], label='y0')
                plt.plot(ts, y1[:, 0, 0], label='y1')
                plt.plot(ts, y2[:, 0, 0], label='y2')
                plt.legend()
                plt.savefig(output_directory / 'plot.png')
                plt.show()
        return

    generate_ode(pathlib.Path('tests/data/ode/interim/train'), 100)
    generate_ode(pathlib.Path('tests/data/ode/interim/validation'), 2)
    generate_ode(pathlib.Path('tests/data/ode/interim/test'), 2)

    main_setting = setting.MainSetting.read_settings_yaml(
        pathlib.Path('tests/data/ode/data.yml'))
    preprocessor = prepost.Preprocessor(main_setting, force_renew=True)
    preprocessor.preprocess_interim_data()
    return


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
    generate_ode()
    preprocess_deform()
    preprocess_deform_timeseries()
    preprocess_linear()
    generate_large()

from pathlib import Path
import shutil

# import matplotlib.pyplot as plt
import numpy as np


DELTA_T = .1


def generate_data(time_length):
    t = np.arange(0., time_length * DELTA_T, DELTA_T)[:, None, None]
    x1 = np.sin((np.random.rand() * .5 + .5) * t + np.random.rand())
    x2 = (np.random.rand() * .5 + .5) * (
        1 - np.exp(- (t + np.random.rand()) / (np.random.rand() + 5.)))
    y1_initial = np.random.rand()
    y2_initial = np.random.rand()

    x3 = np.ones(t.shape) * y1_initial
    x4 = np.ones(t.shape) * y2_initial
    x = np.concatenate([t, x1, x2, x3, x4], axis=2)

    y1 = np.zeros((len(x), 1, 1))
    y1[0] = y1_initial
    for i in range(len(x) - 1):
        y1[i + 1] = y1[i] + DELTA_T * (x[i, 0, 1] - 0.1 * x[i, 0, 2]) * .1

    y2 = np.zeros((len(x), 1, 1))
    y2[0] = y2_initial
    for i in range(len(x) - 1):
        y2[i + 1] = y2[i] + DELTA_T * (
            - 0.3 * x[i, 0, 1] - 0.5 * x[i, 0, 2]) * .1

    y = np.concatenate([y1, y2], axis=2)
    return x, y


def main():

    output_root = Path('tests/data/simplified_timeseries/preprocessed')

    if output_root.exists():
        shutil.rmtree(output_root)

    range_time_length = (100, 200)

    n_train_data = 100
    for i in range(n_train_data):
        time_length = np.random.randint(*range_time_length)
        x, y = generate_data(time_length)

        # plt.plot(x[:, 0, 0], x[:, 0, 1])
        # plt.plot(x[:, 0, 0], x[:, 0, 2])
        # plt.plot(x[:, 0, 0], y[:, 0, 0], label='y1')
        # plt.plot(x[:, 0, 0], y[:, 0, 1], label='y2')
        # plt.legend()
        # plt.show()

        output_directory = output_root / f"train/{i}"
        output_directory.mkdir(parents=True)
        np.save(output_directory / 'x.npy', x.astype(np.float32))
        np.save(output_directory / 'y.npy', y.astype(np.float32))

    n_validation_data = 10
    for i in range(n_validation_data):
        time_length = np.random.randint(*range_time_length)
        x, y = generate_data(time_length)

        output_directory = output_root / f"validation/{i}"
        output_directory.mkdir(parents=True)
        np.save(output_directory / 'x.npy', x.astype(np.float32))
        np.save(output_directory / 'y.npy', y.astype(np.float32))


if __name__ == '__main__':
    main()

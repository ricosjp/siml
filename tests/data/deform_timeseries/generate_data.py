from pathlib import Path
import shutil

# import matplotlib.pyplot as plt
import numpy as np

import siml.prepost as prepost
import siml.util as util


DELTA_T = .1


def main():

    reference_directory = Path('tests/data/deform/preprocessed/train')
    output_root = Path('tests/data/deform_timeseries')

    if (output_root / 'preprocessed').exists():
        shutil.rmtree(output_root / 'preprocessed')

    range_time_length = (100, 200)

    reference_directories = util.collect_data_directories(
        reference_directory, required_file_names=['nadj.npz'])
    for reference_directory in reference_directories:

        # Generate data
        time_length = np.random.randint(*range_time_length)
        scale = np.random.rand()
        scales = np.linspace(0., scale, time_length)

        strain = np.load(reference_directory / 'elemental_strain.npy')
        time_strain = np.stack([s * strain for s in scales])
        stress = np.load(reference_directory / 'elemental_stress.npy')
        time_stress = np.stack([s * stress for s in scales])
        modulus = np.load(reference_directory / 'modulus.npy')
        time_modulus = np.stack([modulus for _ in scales])

        n_element = strain.shape[-2]

        t = np.arange(0., time_length * DELTA_T, DELTA_T)[:, None]
        time_time = np.stack([t for _ in range(n_element)], axis=1)

        # Write files
        output_directory = prepost.determine_output_directory(
            reference_directory, output_root, 'deform')
        output_directory.mkdir(parents=True)
        shutil.copyfile(
            reference_directory / 'nadj.npz', output_directory / 'nadj.npz')
        np.save(output_directory / 't.npy', time_time.astype(np.float32))
        np.save(
            output_directory / 'strain.npy', time_strain.astype(np.float32))
        np.save(
            output_directory / 'stress.npy', time_stress.astype(np.float32))
        np.save(
            output_directory / 'modulus.npy', time_modulus.astype(np.float32))


if __name__ == '__main__':
    main()

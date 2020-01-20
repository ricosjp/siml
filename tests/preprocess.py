import pathlib

import numpy as np
import siml.prepost as prepost
import siml.setting as setting


def conversion_function(fem_data, data_directory):
    adj, _ = fem_data.calculate_adjacency_matrix_element()
    nadj = prepost.normalize_adjacency_matrix(adj)
    global_modulus = np.mean(
        fem_data.access_attribute('modulus'), keepdims=True)
    return {'adj': adj, 'nadj': nadj, 'global_modulus': global_modulus}


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


if __name__ == '__main__':
    preprocess_deform()
    preprocess_deform_timeseries()
    preprocess_linear()

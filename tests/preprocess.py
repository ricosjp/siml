import sys
sys.path.insert(0, '..')  # NOQA

from siml import prepost


def conversion_function(fem_data, data_directory):
    adj, _ = fem_data.calculate_adjacency_matrix_element()
    nadj = prepost.normalize_adjacency_matrix(adj)
    return {'adj': adj, 'nadj': nadj}


def preprocess_deform():
    p = prepost.Preprocessor.read_settings('tests/data/deform/data.yml')
    prepost.convert_raw_data(
        p.setting.data.raw, output_base_directory=p.setting.data.interim,
        mandatory_variables=[
            'elemental_strain', 'modulus',
            'poisson_ratio', 'elemental_stress'],
        recursive=True, force_renew=True,
        conversion_function=conversion_function)
    p.preprocess_interim_data(force_renew=True)


def preprocess_linear():
    p = prepost.Preprocessor.read_settings('tests/data/linear/linear.yml')
    p.preprocess_interim_data(force_renew=True)


if __name__ == '__main__':
    preprocess_deform()
    preprocess_linear()

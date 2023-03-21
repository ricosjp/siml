import numpy as np
import torch
import sys
import os
import pandas as pd
import pytest


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from training_functions import get_output_directory  # NOQA


def test__reproducibility_when_seed_is_same():
    """
    Check reproducibility over several processes
     if random seeds are the same values.
    """
    output_dir_0 = get_output_directory(11, 0)
    output_dir_1 = get_output_directory(11, 1)

    init_seed_0 = torch.load(output_dir_0 / "init_seed.pt")
    init_seed_1 = torch.load(output_dir_1 / "init_seed.pt")

    # Confirm two process is completely different
    assert init_seed_0 != init_seed_1

    df_0 = pd.read_csv(
        output_dir_0 / "log.csv",
        header=0,
        index_col=None,
        skipinitialspace=True
    )

    df_1 = pd.read_csv(
        output_dir_1 / "log.csv",
        header=0,
        index_col=None,
        skipinitialspace=True
    )

    for column in ['train_loss', 'validation_loss']:
        array_0 = df_0.loc[:, column].to_numpy()
        array_1 = df_1.loc[:, column].to_numpy()
        np.testing.assert_array_equal(
            array_0,
            array_1
        )


def test__reproducibility_when_seed_is_diffrenet():
    """
    Check reproducing different results over several processes
     if random seeds are different.
    """
    output_dir_0 = get_output_directory(11, 0)
    output_dir_1 = get_output_directory(23, 0)

    init_seed_0 = torch.load(output_dir_0 / "init_seed.pt")
    init_seed_1 = torch.load(output_dir_1 / "init_seed.pt")

    # Confirm two process is completely different
    assert init_seed_0 != init_seed_1

    df_0 = pd.read_csv(
        output_dir_0 / "log.csv",
        header=0,
        index_col=None,
        skipinitialspace=True
    )

    df_1 = pd.read_csv(
        output_dir_1 / "log.csv",
        header=0,
        index_col=None,
        skipinitialspace=True
    )

    for column in ['train_loss', 'validation_loss']:
        array_0 = df_0.loc[:, column].to_numpy()
        array_1 = df_1.loc[:, column].to_numpy()
        with pytest.raises(AssertionError):
            np.testing.assert_equal(
                array_0,
                array_1
            )

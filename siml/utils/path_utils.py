import os
from pathlib import Path

import numpy as np


def determine_output_directory(
    input_directory: Path,
    output_base_directory: Path,
    str_replace: str
) -> Path:
    """Determine output directory by replacing a string (str_replace) in the
    input_directory.

    Parameters
    ----------
    input_directory: pathlib.Path
        Input directory path.
    output_base_directory: pathlib.Path
        Output base directory path. The output directry name is under that
        directory.
    str_replace: str
        The string to be replaced.

    Returns
    -------
    output_directory: pathlib.Path
        Detemined output directory path.
    """
    common_prefix = common_parent(
        input_directory,
        output_base_directory
    )
    relative_input_path = Path(os.path.relpath(input_directory, common_prefix))
    parts = list(relative_input_path.parts)

    replace_indices = np.where(
        np.array(relative_input_path.parts) == str_replace)[0]
    if len(replace_indices) == 0:
        pass
    elif len(replace_indices) == 1:
        replace_index = replace_indices[0]
        parts[replace_index] = ''
    else:
        raise ValueError(
            f"Input directory {input_directory} contains several "
            f"{str_replace} parts thus ambiguous.")
    output_directory = output_base_directory / '/'.join(parts).lstrip('/')

    return output_directory


def common_parent(
    directory_1: Path,
    directory_2: Path
) -> Path:
    """Search common parent directory

    Parameters
    ----------
    directory_1 : pathlib.Path
    directory_2 : pathlib.Path

    Returns
    -------
    common_parent: pathlib.Path
        Path to common parent directory
    """
    parents_1 = directory_1.parents
    parents_2 = directory_2.parents
    min_idx_1 = len(parents_1) - 1
    min_idx_2 = len(parents_2) - 1
    min_idx = min(len(parents_1), len(parents_2))

    common_parent = Path("")
    for i in range(min_idx):
        if parents_1[min_idx_1 - i] == parents_2[min_idx_2 - i]:
            common_parent = parents_1[min_idx_1 - i]
        else:
            break
    return common_parent

import pathlib
from typing import NamedTuple, Optional, Union

import femio
import numpy as np

from siml.siml_variables import ISimlVariables


class RawPredictionRecord(NamedTuple):
    y_pred: ISimlVariables
    y: ISimlVariables
    x: ISimlVariables
    original_shapes: tuple
    inference_time: float
    data_directory: Union[pathlib.Path, None]


class PredictionRecord(NamedTuple):
    dict_x: dict[str, np.ndarray]
    dict_y: dict[str, np.ndarray]
    original_shapes: tuple
    data_directory: pathlib.Path
    inference_time: float
    inference_start_datetime: str
    dict_answer: Optional[dict[str, np.ndarray]] = None


class PostPredictionRecord(NamedTuple):
    dict_x: dict[str, np.ndarray]
    dict_y: dict[str, np.ndarray]
    original_shapes: tuple
    data_directory: pathlib.Path
    inference_time: float
    inference_start_datetime: str
    dict_answer: dict[str, np.ndarray]
    # above items must be the same order of PredictionRecord
    loss: float
    raw_loss: float
    output_directory: pathlib.Path
    fem_data: Optional[femio.FEMData] = None

from typing import Optional
import pathlib

from torch.utils.data import DataLoader

from siml import datasets, setting


class InferenceDataLoaderBuilder():
    def __init__(
        self,
        trainer_setting: setting.TrainerSetting,
        collate_fn: datasets.CollateFunctionGenerator,
        decrypt_key: Optional[bytes] = None
    ) -> None:
        self._trainer_setting = trainer_setting
        self._collate_fn = collate_fn
        self._decrypt_key = decrypt_key

    def create(
        self,
        data_directories: list[pathlib.Path] = None,
        raw_dict_x: dict = None,
        answer_raw_dict_y: dict = None,
        allow_no_data: bool = True
    ) -> DataLoader:

        if raw_dict_x is not None:
            inference_dataset = datasets.SimplifiedDataset(
                self._trainer_setting.input_names,
                self._trainer_setting.output_names,
                raw_dict_x=raw_dict_x,
                answer_raw_dict_y=answer_raw_dict_y,
                allow_no_data=allow_no_data,
                num_workers=0,
                supports=self._trainer_setting.support_inputs,
                directories=data_directories
            )
        else:
            inference_dataset = datasets.LazyDataset(
                self._trainer_setting.input_names,
                self._trainer_setting.output_names,
                data_directories,
                supports=self._trainer_setting.support_inputs,
                num_workers=0,
                allow_no_data=allow_no_data,
                decrypt_key=self._decrypt_key
            )

        inference_loader = DataLoader(
            inference_dataset,
            collate_fn=self._collate_fn,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        return inference_loader

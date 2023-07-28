import pathlib
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

from siml import util
from siml import datasets, setting
from siml.path_like_objects import SimlDirectory


class DataLoaderBuilder():
    def __init__(
        self,
        main_setting: setting.MainSetting,
        collate_fn: datasets.CollateFunctionGenerator,
        decrypt_key: Optional[bytes] = None
    ) -> None:
        self._trainer_setting = main_setting.trainer
        self._data_setting = main_setting.data
        self._collate_fn = collate_fn
        self._decrypt_key = decrypt_key

    def _set_dataset_directories(self):
        if len(self._trainer_setting.split_ratio) == 0:
            return

        develop_dataset = datasets.LazyDataset(
            self._trainer_setting.input_names,
            self._trainer_setting.output_names,
            self._data_setting.develop,
            recursive=self._trainer_setting.recursive,
            decrypt_key=self._data_setting.encrypt_key)

        # HACK: Maybe not necessary to create
        train, validation, test = util.split_data(
            develop_dataset.data_directories,
            **self._trainer_setting.split_ratio)
        self._data_setting.train = train
        self._data_setting.validation = validation
        self._data_setting.test = test

    def create(
        self
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create dataloaders

        Returns
        -------
        tuple[DataLoader, DataLoader, DataLoader]
            Train loader, validaition loader, test loader
        """

        self._set_dataset_directories()
        batch_size, validation_batch_size = \
            self._trainer_setting.determine_batch_sizes()
        element_wise = self._trainer_setting.determine_element_wise()
        train_directories = self._data_setting.train
        validation_directories = self._data_setting.validation
        test_directories = self._data_setting.test

        if self._trainer_setting.lazy:
            return self._get_data_loaders(
                datasets.LazyDataset,
                batch_size,
                validation_batch_size=validation_batch_size,
                train_directories=train_directories,
                validation_directories=validation_directories,
                test_directories=test_directories,
                decrypt_key=self._decrypt_key
            )

        if element_wise:
            return self._get_data_loaders(
                datasets.ElementWiseDataset,
                batch_size,
                validation_batch_size=validation_batch_size,
                train_directories=train_directories,
                validation_directories=validation_directories,
                test_directories=test_directories,
                decrypt_key=self._decrypt_key
            )

        return self._get_data_loaders(
            datasets.OnMemoryDataset,
            batch_size,
            validation_batch_size=validation_batch_size,
            train_directories=train_directories,
            validation_directories=validation_directories,
            test_directories=test_directories,
            decrypt_key=self._decrypt_key
        )

    def _get_data_loaders(
        self,
        dataset_generator: Dataset,
        batch_size: int,
        train_directories: list[pathlib.Path],
        validation_directories: list[pathlib.Path],
        test_directories: list[pathlib.Path],
        validation_batch_size: Optional[int] = None,
        decrypt_key: Optional[bytes] = None
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        if validation_batch_size is None:
            validation_batch_size = batch_size

        x_variable_names = self._trainer_setting.input_names
        y_variable_names = self._trainer_setting.output_names
        supports = self._trainer_setting.support_inputs
        num_workers = self._trainer_setting.num_workers
        recursive = self._trainer_setting.recursive

        train_dataset = dataset_generator(
            x_variable_names, y_variable_names,
            train_directories, supports=supports, num_workers=num_workers,
            recursive=recursive, decrypt_key=decrypt_key)
        validation_dataset = dataset_generator(
            x_variable_names, y_variable_names,
            validation_directories, supports=supports, num_workers=num_workers,
            recursive=recursive, allow_no_data=True, decrypt_key=decrypt_key)
        test_dataset = dataset_generator(
            x_variable_names, y_variable_names,
            test_directories, supports=supports, num_workers=num_workers,
            recursive=recursive, allow_no_data=True, decrypt_key=decrypt_key)

        random_generator = torch.Generator()
        random_generator.manual_seed(self._trainer_setting.seed)

        print(f"num_workers for data_loader: {num_workers}")
        train_loader = DataLoader(
            train_dataset,
            collate_fn=self._collate_fn,
            batch_size=batch_size,
            shuffle=self._trainer_setting.train_data_shuffle,
            num_workers=num_workers,
            worker_init_fn=self._seed_worker,
            generator=random_generator
        )
        validation_loader = DataLoader(
            validation_dataset,
            collate_fn=self._collate_fn,
            batch_size=validation_batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=self._seed_worker,
            generator=random_generator
        )
        test_loader = DataLoader(
            test_dataset,
            collate_fn=self._collate_fn,
            batch_size=validation_batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=self._seed_worker,
            generator=random_generator
        )

        self._check_data_dimension(
            train_loader,
            self._trainer_setting.input_names_list
        )

        self._check_data_dimension(
            train_loader,
            self._trainer_setting.output_names_list
        )

        return train_loader, validation_loader, test_loader

    def _seed_worker(worker_id) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _check_data_dimension(
        self,
        data_loader: DataLoader,
        variable_names: list[str]
    ) -> None:

        data_directory = data_loader.dataset.data_directories[0]

        for variable_name in variable_names:
            self._check_single_variable_dimension(
                data_directory,
                variable_name
            )

    def _check_single_variable_dimension(
        self,
        data_directory: pathlib.Path,
        variable_name: str
    ):

        siml_dir = SimlDirectory(data_directory)
        siml_file = siml_dir.find_variable_file(variable_name)
        loaded_data = siml_file.load(decrypt_key=self._decrypt_key)

        shape = loaded_data.shape
        variable_info = \
            self._trainer_setting.variable_information[variable_name]

        if len(shape) == 2:
            if shape[-1] != variable_info['dim']:
                raise ValueError(
                    f"{variable_name} dimension incorrect: "
                    f"{shape} vs {variable_info['dim']}"
                )
        return

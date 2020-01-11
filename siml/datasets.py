
from ignite.utils import convert_tensor
import numpy as np
import torch

from . import util


class BaseDataset(torch.utils.data.Dataset):

    def __init__(
            self, x_variable_names, y_variable_names, directories, *,
            supports=None):
        self.x_variable_names = x_variable_names
        self.y_variable_names = y_variable_names
        self.supports = supports

        data_directories = []
        for directory in directories:
            data_directories += util.collect_data_directories(
                directory, required_file_names=[f"{x_variable_names[0]}.npy"])
        self.data_directories = np.unique(data_directories)

        if len(self) == 0:
            raise ValueError(f"No data dound in {directories}")

        return

    def __len__(self):
        return len(self.data_directories)

    def _load_data(self, data_directory):
        x_data = util.concatenate_variable([
            util.load_variable(data_directory, x_variable_name)
            for x_variable_name in self.x_variable_names])
        y_data = util.concatenate_variable([
            util.load_variable(data_directory, y_variable_name)
            for y_variable_name in self.y_variable_names])
        if self.supports is None:
            return {
                'x': torch.from_numpy(x_data), 't': torch.from_numpy(y_data)}
        else:
            # TODO: use appropreate sparse data class
            support_data = [
                util.load_variable(data_directory, support)
                for support in self.supports]
            return {
                'x': torch.from_numpy(x_data), 't': torch.from_numpy(y_data),
                'supports': support_data}


class LazyDataset(BaseDataset):

    def __getitem__(self, i):
        data_directory = self.data_directories[i]
        return self._load_data(data_directory)


class OnMemoryDataset(BaseDataset):

    def __init__(
            self, x_variable_names, y_variable_names, directories, *,
            supports=None):
        super().__init__(
            x_variable_names, y_variable_names, directories, supports=supports)
        print(f"Loading data for: {directories}")
        self.data = [
            self._load_data(data_directory)
            for data_directory in self.data_directories]

    def __len__(self):
        return len(self.data_directories)

    def __getitem__(self, i):
        return self.data[i]


def prepare_batch_with_support(batch, device=None, non_blocking=False):
    return (
        (
            convert_tensor(
                batch['x'], device=device, non_blocking=non_blocking),
            convert_tensor(
                batch['supports'], device=device, non_blocking=non_blocking),
        ),
        convert_tensor(
            batch['t'], device=device, non_blocking=non_blocking))


def prepare_batch_without_support(batch, device=None, non_blocking=False):
    return (
        (
            convert_tensor(
                batch['x'], device=device, non_blocking=non_blocking),
        ),
        convert_tensor(
            batch['t'], device=device, non_blocking=non_blocking))

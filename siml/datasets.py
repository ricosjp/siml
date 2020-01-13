
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

        if len(self.data_directories) == 0:
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


class ElementWiseDataset(BaseDataset):

    def __init__(
            self, x_variable_names, y_variable_names, directories, *,
            supports=None):
        super().__init__(
            x_variable_names, y_variable_names, directories, supports=supports)
        print(f"Loading data for: {directories}")
        loaded_data = [
            self._load_data(data_directory)
            for data_directory in self.data_directories]
        self.x = np.concatenate([ld['x'] for ld in loaded_data])
        self.t = np.concatenate([ld['t'] for ld in loaded_data])
        return

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {
            'x': torch.from_numpy(self.x[i]), 't': torch.from_numpy(self.t[i])}


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
        return

    def __getitem__(self, i):
        return self.data[i]


def collate_fn_with_support(batch):
    x = pad_dense_sequence(batch, 'x')
    t = concatenate_sequence(batch, 't')
    length = x.shape[1]
    padded_supports = pad_sparse_sequence(batch, 'supports', length)

    original_lengths = [b['x'].shape[0] for b in batch]

    return {
        'x': x, 't': t, 'supports': padded_supports,
        'original_lengths': original_lengths}


def collate_fn_without_support(batch):
    x = pad_dense_sequence(batch, 'x')
    t = concatenate_sequence(batch, 't')

    original_lengths = [b['x'].shape[0] for b in batch]

    return {'x': x, 't': t, 'original_lengths': original_lengths}


def concatenate_sequence(batch, key):
    return torch.cat([b[key] for b in batch])


def collate_fn_element_wise(batch):
    x = stack_sequence(batch, 'x')
    t = stack_sequence(batch, 't')
    return {'x': x, 't': t, 'original_lengths': None}


def stack_sequence(batch, key):
    return torch.stack([b[key] for b in batch])


def pad_dense_sequence(batch, key):
    return torch.nn.utils.rnn.pad_sequence(
        [b[key] for b in batch], batch_first=True)


def pad_sparse_sequence(batch, key, length):
    return [[pad_sparse(s, length) for s in b[key]] for b in batch]


def pad_sparse(sparse, length):
    """Pad sparse matrix.

    Parameters
    ----------
    sparse: scipy.sparse.coo_matrix
    length: int

    Returns
    -------
    padded_sparse: dict
        NOTE: So far dict is returned due to the lack of DataLoader support for
        sparse tensor https://github.com/pytorch/pytorch/issues/20248 .
        The dict will be converted to the sparse tensor at the timing of
        prepare_batch is called.
    """
    indices = torch.LongTensor([sparse.row, sparse.col])
    values = torch.from_numpy(sparse.data)

    return {
        'indices': indices, 'values': values,
        'size': torch.Size((length, length))}


def prepare_batch_with_support(batch, device=None, non_blocking=False):
    return {
        'x': convert_tensor(
            batch['x'], device=device, non_blocking=non_blocking),
        'supports': convert_sparse_tensor(
            batch['supports'], device=device, non_blocking=non_blocking),
        'original_lengths': batch['original_lengths'],
    }, convert_tensor(batch['t'], device=device, non_blocking=non_blocking)


def convert_sparse_tensor(sparse_info, device=None, non_blocking=False):
    return [
        [
            torch.sparse_coo_tensor(
                s['indices'], s['values'], s['size']).to(device)
            for s in si]
        for si in sparse_info]


def prepare_batch_without_support(batch, device=None, non_blocking=False):
    return {
        'x': convert_tensor(
            batch['x'], device=device, non_blocking=non_blocking),
        'original_lengths': batch['original_lengths'],
    }, convert_tensor(batch['t'], device=device, non_blocking=non_blocking)

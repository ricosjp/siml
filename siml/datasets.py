import multiprocessing as multi

from ignite.utils import convert_tensor
import numpy as np
import torch
from tqdm import tqdm

from . import util


class BaseDataset(torch.utils.data.Dataset):

    def __init__(
            self, x_variable_names, y_variable_names, directories, *,
            supports=None, num_workers=0):
        self.x_variable_names = x_variable_names
        self.y_variable_names = y_variable_names
        self.supports = supports
        self.num_workers = num_workers

        data_directories = []
        for directory in directories:
            data_directories += util.collect_data_directories(
                directory, required_file_names=[f"{x_variable_names[0]}.npy"])
        self.data_directories = np.unique(data_directories)

        if len(self.data_directories) == 0:
            raise ValueError(f"No data found in {directories}")

        return

    def __len__(self):
        return len(self.data_directories)

    def _load_all_data(self, data_directories):
        print('Loading data')

        # if self.num_workers < 1:
        if True:
            # Single process
            pbar = tqdm(
                initial=0, leave=False, total=len(data_directories),
                ncols=80, ascii=True)
            data = [
                self._load_data(data_directory, pbar)
                for data_directory in data_directories]
        else:
            # TODO: After upgrading to Python3.8, activate this block to
            # communicate large data
            # https://stackoverflow.com/questions/47776486/python-struct-error-i-format-requires-2147483648-number-2147483647  # NOQA
            # https://github.com/python/cpython/pull/10305
            chunksize = max(len(data_directories) // self.num_workers // 16, 1)
            with multi.Pool(self.num_workers) as pool:
                data = list(tqdm(
                    pool.imap(
                        self._load_data, data_directories,
                        chunksize=chunksize),
                    initial=0, leave=False, total=len(data_directories),
                    ncols=80, ascii=True))

        pbar.close()
        return data

    def _load_data(self, data_directory, pbar=None):
        x_data = util.concatenate_variable([
            util.load_variable(data_directory, x_variable_name)
            for x_variable_name in self.x_variable_names])
        y_data = util.concatenate_variable([
            util.load_variable(data_directory, y_variable_name)
            for y_variable_name in self.y_variable_names])
        if pbar:
            pbar.update()
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
            supports=None, num_workers=0):
        super().__init__(
            x_variable_names, y_variable_names, directories, supports=supports,
            num_workers=num_workers)

        loaded_data = self._load_all_data(self.data_directories)

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
            supports=None, num_workers=0):
        super().__init__(
            x_variable_names, y_variable_names, directories, supports=supports,
            num_workers=num_workers)

        self.data = self._load_all_data(self.data_directories)
        return

    def __getitem__(self, i):
        return self.data[i]


def collate_fn_with_support(batch):
    x = pad_dense_sequence(batch, 'x')
    t = concatenate_sequence(batch, 't')
    max_element_length = x.shape[-2]
    padded_supports = pad_sparse_sequence(
        batch, 'supports', max_element_length)

    original_shapes = [b['x'].shape[:1] for b in batch]

    return {
        'x': x, 't': t, 'supports': padded_supports,
        'original_shapes': original_shapes}


def collate_fn_time_with_support(batch):
    x = pad_time_dense_sequence(batch, 'x')
    t = pad_time_dense_sequence(batch, 't')
    max_element_length = x.shape[-2]
    padded_supports = pad_sparse_sequence(
        batch, 'supports', max_element_length)

    original_shapes = [[b['x'].shape[0], b['x'].shape[-2]] for b in batch]

    return {
        'x': x, 't': t, 'supports': padded_supports,
        'original_shapes': original_shapes}


def collate_fn_without_support(batch):
    x = pad_dense_sequence(batch, 'x')
    t = concatenate_sequence(batch, 't')

    original_shapes = [b['x'].shape[:1] for b in batch]

    return {'x': x, 't': t, 'original_shapes': original_shapes}


def collate_fn_time_without_support(batch):
    x = pad_time_dense_sequence(batch, 'x')
    t = pad_time_dense_sequence(batch, 't')

    original_shapes = [[b['x'].shape[0], b['x'].shape[-2]] for b in batch]

    return {
        'x': x, 't': t, 'original_shapes': original_shapes}


def concatenate_sequence(batch, key):
    return torch.cat([b[key] for b in batch])


def collate_fn_element_wise(batch):
    x = stack_sequence(batch, 'x')
    t = stack_sequence(batch, 't')
    return {'x': x, 't': t, 'original_shapes': None}


def stack_sequence(batch, key):
    return torch.stack([b[key] for b in batch])


def pad_dense_sequence(batch, key):
    return torch.nn.utils.rnn.pad_sequence(
        [b[key] for b in batch], batch_first=True)


def pad_time_dense_sequence(batch, key):
    data = [b[key] for b in batch]

    max_time_lengths = np.max([d.shape[0] for d in data])
    time_padded_data = [
        pad_time_direction(d, max_time_lengths) for d in data]
    padded_data = torch.nn.utils.rnn.pad_sequence(
        [d.permute(1, 0, 2) for d in time_padded_data],
        batch_first=True).permute(2, 0, 1, 3)
    return padded_data


def pad_time_direction(data, time_length):
    if len(data) == time_length:
        return data
    remaining_length = time_length - len(data)
    zeros = torch.zeros((remaining_length, *data.shape[1:]))
    padded_data = torch.cat([data, zeros])
    return padded_data


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
    row = torch.LongTensor(sparse.row)
    col = torch.LongTensor(sparse.col)
    values = torch.from_numpy(sparse.data)

    return {
        'row': row, 'col': col, 'values': values,
        'size': torch.Size((length, length))}


def prepare_batch_with_support(
        batch, device=None, output_device=None, non_blocking=False):
    return {
        'x': convert_tensor(
            batch['x'], device=device, non_blocking=non_blocking),
        'supports': convert_sparse_info(
            batch['supports'], device=device, non_blocking=non_blocking),
        'original_shapes': batch['original_shapes'],
    }, convert_tensor(
        batch['t'], device=output_device, non_blocking=non_blocking)


def convert_sparse_info(sparse_info, device=None, non_blocking=False):
    device = 'cpu'
    return [
        [{
            'row': convert_tensor(
                s['row'], device=device, non_blocking=non_blocking),
            'col': convert_tensor(
                s['col'], device=device, non_blocking=non_blocking),
            'values': convert_tensor(
                s['values'], device=device, non_blocking=non_blocking),
            'size': s['size'],
        } for s in si] for si in sparse_info]


def convert_sparse_tensor(sparse_info, device=None, non_blocking=False):
    return np.array([
        [
            torch.sparse_coo_tensor(
                torch.stack([s['row'], s['col']]),
                s['values'], s['size']
            ).to(device)
            for s in si]
        for si in sparse_info])


def prepare_batch_without_support(
        batch, device=None, output_device=None, non_blocking=False):
    return {
        'x': convert_tensor(
            batch['x'], device=device, non_blocking=non_blocking),
        'original_shapes': batch['original_shapes'],
    }, convert_tensor(
        batch['t'], device=output_device, non_blocking=non_blocking)

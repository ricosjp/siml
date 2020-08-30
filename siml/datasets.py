import multiprocessing as multi

from ignite.utils import convert_tensor
import numpy as np
import torch
from tqdm import tqdm

from . import util


class BaseDataset(torch.utils.data.Dataset):

    def __init__(
            self, x_variable_names, y_variable_names, directories, *,
            supports=None, num_workers=0, allow_no_data=False):
        self.x_variable_names = x_variable_names
        self.y_variable_names = y_variable_names
        self.supports = supports
        self.num_workers = num_workers

        self.x_dict_mode = isinstance(self.x_variable_names, dict)
        self.y_dict_mode = isinstance(self.y_variable_names, dict)

        if self.x_dict_mode:
            first_variable_name = list(self.x_variable_names.values())[0][0]
        else:
            first_variable_name = self.x_variable_names[0]

        data_directories = []
        for directory in directories:
            data_directories += util.collect_data_directories(
                directory, required_file_names=[f"{first_variable_name}.npy"],
                allow_no_data=allow_no_data)
        self.data_directories = np.unique(data_directories)

        if not allow_no_data and len(self.data_directories) == 0:
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

    def _load_from_names(self, data_directory, variable_names):
        if isinstance(variable_names, dict):
            return {
                key: torch.from_numpy(util.concatenate_variable([
                    util.load_variable(data_directory, variable_name)
                    for variable_name in value]))
                for key, value in variable_names.items()}
        elif isinstance(variable_names, list):
            return torch.from_numpy(util.concatenate_variable([
                util.load_variable(data_directory, variable_name)
                for variable_name in variable_names]))
        else:
            raise ValueError(f"Unexpected variable names: {variable_names}")

    def _load_data(self, data_directory, pbar=None):
        x_data = self._load_from_names(
            data_directory, self.x_variable_names)
        y_data = self._load_from_names(
            data_directory, self.y_variable_names)

        if pbar:
            pbar.update()
        if self.supports is None:
            return {'x': x_data, 't': y_data}
        else:
            # TODO: use appropreate sparse data class
            support_data = [
                util.load_variable(data_directory, support)
                for support in self.supports]
            return {'x': x_data, 't': y_data, 'supports': support_data}


class LazyDataset(BaseDataset):

    def __getitem__(self, i):
        data_directory = self.data_directories[i]
        return self._load_data(data_directory)


class ElementWiseDataset(BaseDataset):

    def __init__(
            self, x_variable_names, y_variable_names, directories, *,
            supports=None, num_workers=0, allow_no_data=False):
        super().__init__(
            x_variable_names, y_variable_names, directories, supports=supports,
            num_workers=num_workers, allow_no_data=allow_no_data)

        if len(self.data_directories) == 0:
            self.x = []
            self.t = []
        else:
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
            supports=None, num_workers=0, allow_no_data=False):
        super().__init__(
            x_variable_names, y_variable_names, directories, supports=supports,
            num_workers=num_workers, allow_no_data=allow_no_data)

        self.data = self._load_all_data(self.data_directories)
        return

    def __getitem__(self, i):
        return self.data[i]


def collate_fn_with_support(batch):
    x = concatenate_sequence(batch, 'x')
    t = concatenate_sequence(batch, 't')
    padded_supports = concatenate_sparse_sequence(batch, 'supports')

    original_shapes = np.array(
        [[b['x'].shape[0]] for b in batch])
    return {
        'x': x, 't': t, 'supports': padded_supports,
        'original_shapes': original_shapes}


def collate_fn_time_with_support(batch):
    x = pad_time_dense_sequence(batch, 'x')
    t = pad_time_dense_sequence(batch, 't')
    padded_supports = concatenate_sparse_sequence(batch, 'supports')

    original_shapes = np.array(
        [[b['x'].shape[0], b['x'].shape[-2]] for b in batch])
    return {
        'x': x, 't': t, 'supports': padded_supports,
        'original_shapes': original_shapes}


def collate_fn_without_support(batch):
    x = concatenate_sequence(batch, 'x')
    t = concatenate_sequence(batch, 't')

    original_shapes = np.array(
        [[b['x'].shape[0]] for b in batch])

    return {'x': x, 't': t, 'original_shapes': original_shapes}


def collate_fn_time_without_support(batch):
    x = pad_time_dense_sequence(batch, 'x')
    t = pad_time_dense_sequence(batch, 't')

    original_shapes = np.array(
        [[b['x'].shape[0], b['x'].shape[-2]] for b in batch])
    return {
        'x': x, 't': t, 'original_shapes': original_shapes}


def concatenate_sequence(batch, key):
    if isinstance(batch[0][key], dict):
        return {
            dict_key:
            torch.cat([b[key][dict_key] for b in batch])
            for dict_key in batch[0][key].keys()}
    else:
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
    return torch.cat([
        pad_time_direction(d, max_time_lengths) for d in data], axis=1)


def pad_time_direction(data, time_length):
    if len(data) == time_length:
        return data
    remaining_length = time_length - len(data)
    zeros = torch.zeros((remaining_length, *data.shape[1:]))
    padded_data = torch.cat([data, zeros])
    return padded_data


def concatenate_sparse_sequence(batch, key):
    sparse_infos = pad_sparse_sequence(batch, key)
    n_sparse_features = len(sparse_infos[0])
    return [
        merge_sparse_tensors([s[i] for s in sparse_infos], return_coo=False)
        for i in range(n_sparse_features)]


def pad_sparse_sequence(batch, key, length=None):
    return [[pad_sparse(s, length) for s in b[key]] for b in batch]


def pad_sparse(sparse, length=None):
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
    if length is None:
        shape = sparse.shape
    else:
        shape = (length, length)

    return {
        'row': row, 'col': col, 'values': values,
        'size': torch.Size(shape)}


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
    if isinstance(sparse_info[0], list):
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
    else:
        return [{
            'row': convert_tensor(
                si['row'], device=device, non_blocking=non_blocking),
            'col': convert_tensor(
                si['col'], device=device, non_blocking=non_blocking),
            'values': convert_tensor(
                si['values'], device=device, non_blocking=non_blocking),
            'size': si['size'],
        } for si in sparse_info]


def convert_sparse_tensor(
        sparse_info, device=None, non_blocking=False, merge=False):
    """Convert sparse info to torch.Tensor which is sparse.

    Parameters
    ----------
    sparse_info: List[List[Dict[str: torch.Tensor]]]
        Sparse data which has: row, col, values, size in COO format.
    non_blocking: bool, optional
        Dummy parameter to have unified interface with
        ignite.utils.convert_tensor.
    merge: bool, optional
        If True, create large sparse tensor merged in the diag direction.

    Returns
    -------
    sparse_tensors: numpy.ndarray[torch.Tensor]
    """
    if merge:
        converted_sparses = np.array([
            merge_sparse_tensors(si).to(device)
            for si in sparse_info])
    else:
        if isinstance(sparse_info[0], list):
            converted_sparses = np.array([
                [
                    torch.sparse_coo_tensor(
                        torch.stack([s['row'], s['col']]),
                        s['values'], s['size']
                    ).to(device)
                    for s in si]
                for si in sparse_info])
        else:
            converted_sparses = np.array([
                torch.sparse_coo_tensor(
                    torch.stack([si['row'], si['col']]),
                    si['values'], si['size']
                ).to(device)
                for si in sparse_info])

    return converted_sparses


def merge_sparse_tensors(stripped_sparse_info, *, return_coo=True):
    """Merge sparse tensors.

    Parameters
    ----------
    stripped_sparse_info: List[Dict[str: torch.Tensor]]
        Sparse data which has: row, col, values, size in COO format.
    return_coo: bool
        If True, return torch.sparse_coo_tensor. Else, return sparse info
        dict. The default is True.

    Returns
    -------
        merged_sparse_tensor: torch.Tensor
    """
    row_shifts = np.cumsum([s['size'][0] for s in stripped_sparse_info])
    col_shifts = np.cumsum([s['size'][1] for s in stripped_sparse_info])
    rows = [s['row'] for s in stripped_sparse_info]
    cols = [s['col'] for s in stripped_sparse_info]
    values = torch.cat([s['values'] for s in stripped_sparse_info])

    merged_rows = rows[0]
    merged_cols = cols[0]
    for i in range(1, len(rows)):
        merged_rows = torch.cat([merged_rows, rows[i] + row_shifts[i-1]])
        merged_cols = torch.cat([merged_cols, cols[i] + col_shifts[i-1]])

    shape = [row_shifts[-1], col_shifts[-1]]
    if return_coo:
        return torch.sparse_coo_tensor(
            torch.stack([merged_rows, merged_cols]), values, shape)
    else:
        return {
            'row': merged_rows, 'col': merged_cols, 'values': values,
            'size': shape}


def prepare_batch_without_support(
        batch, device=None, output_device=None, non_blocking=False):
    return {
        'x': convert_tensor(
            batch['x'], device=device, non_blocking=non_blocking),
        'original_shapes': batch['original_shapes'],
    }, convert_tensor(
        batch['t'], device=output_device, non_blocking=non_blocking)

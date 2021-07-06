import multiprocessing as multi

import femio
from ignite.utils import convert_tensor
import numpy as np
import torch
from tqdm import tqdm

from . import prepost
from . import util


class BaseDataset(torch.utils.data.Dataset):

    def __init__(
            self, x_variable_names, y_variable_names, directories, *,
            supports=None, num_workers=0, allow_no_data=False,
            decrypt_key=None, required_file_names=None, **kwargs):
        self.x_variable_names = x_variable_names
        self.y_variable_names = y_variable_names
        self.supports = supports
        self.num_workers = num_workers
        self.decrypt_key = decrypt_key

        self.x_dict_mode = isinstance(self.x_variable_names, dict)
        self.y_dict_mode = isinstance(self.y_variable_names, dict)

        if self.x_dict_mode:
            first_variable_name = list(self.x_variable_names.values())[0][0]
        else:
            first_variable_name = self.x_variable_names[0]

        if required_file_names is None:
            required_file_names = [f"{first_variable_name}.npy"]

        data_directories = []
        for directory in directories:
            data_directories += util.collect_data_directories(
                directory, required_file_names=required_file_names,
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
            return DataDict({
                key: torch.from_numpy(util.concatenate_variable([
                    util.load_variable(
                        data_directory, variable_name,
                        decrypt_key=self.decrypt_key)
                    for variable_name in value]))
                for key, value in variable_names.items()})
        elif isinstance(variable_names, list):
            return torch.from_numpy(util.concatenate_variable([
                util.load_variable(
                    data_directory, variable_name,
                    decrypt_key=self.decrypt_key)
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
            return {'x': x_data, 't': y_data, 'data_directory': data_directory}
        else:
            # TODO: use appropreate sparse data class
            support_data = [
                util.load_variable(
                    data_directory, support, decrypt_key=self.decrypt_key)
                for support in self.supports]
            return {
                'x': x_data, 't': y_data, 'supports': support_data,
                'data_directory': data_directory}

    def _merge_data(self, converted_dict_data, variable_names):
        if isinstance(variable_names, dict):
            return DataDict({
                key: torch.from_numpy(util.concatenate_variable([
                    converted_dict_data[variable_name]
                    for variable_name in value]).astype(np.float32))
                for key, value in variable_names.items()})
        elif isinstance(variable_names, list):
            return torch.from_numpy(util.concatenate_variable([
                converted_dict_data[variable_name].astype(np.float32)
                for variable_name in variable_names]).astype(
                np.float32))
        else:
            raise ValueError(f"Unexpected variable names: {variable_names}")


class LazyDataset(BaseDataset):

    def __getitem__(self, i):
        data_directory = self.data_directories[i]
        return self._load_data(data_directory)


class PreprocessDataset(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepost_converter = kwargs.pop('prepost_converter')
        self.raw_data_stem = kwargs.pop('raw_data_stem', None)
        self.conversion_function = kwargs.pop('conversion_function', None)
        self.load_function = kwargs.pop('load_function', None)
        self.conversion_setting = kwargs.pop('conversion_setting', None)
        return

    def __getitem__(self, i):
        data_directory = self.data_directories[i]
        return self._preprocess_data(data_directory)

    def _preprocess_data(self, raw_data_directory):
        if self.conversion_setting.skip_femio:
            dict_data = {}
        else:
            fem_data = femio.FEMData.read_directory(
                self.conversion_setting.file_type, raw_data_directory,
                save=False)
            dict_data = prepost.extract_variables(
                fem_data, self.conversion_setting.mandatory,
                optional_variables=self.conversion_setting.optional)

        if self.conversion_function is not None:
            dict_data.update(self.conversion_function(
                fem_data, raw_data_directory))

        if self.load_function is not None:
            data_files = util.collect_files(
                raw_data_directory,
                self.conversion_setting.required_file_names)
            loaded_dict_data, _ = self.load_function(
                data_files, raw_data_directory)
            dict_data.update(loaded_dict_data)

        converted_dict_data = self.prepost_converter.preprocess(dict_data)
        x_data = self._merge_data(
            converted_dict_data, self.x_variable_names)

        if np.all([
                y_variable_names in converted_dict_data
                for y_variable_names in self.y_variable_names]):
            y_data = self._merge_data(
                converted_dict_data, self.y_variable_names)
        else:
            y_data = None

        if self.supports is None:
            return {
                'x': x_data, 't': y_data, 'data_directory': raw_data_directory}
        else:
            support_data = [
                converted_dict_data[support_input_name].astype(np.float32)
                for support_input_name in self.supports]
            return {
                'x': x_data, 't': y_data, 'supports': support_data,
                'data_directory': raw_data_directory}


class ElementWiseDataset(BaseDataset):

    def __init__(
            self, x_variable_names, y_variable_names, directories, *,
            supports=None, num_workers=0, allow_no_data=False, **kwargs):
        super().__init__(
            x_variable_names, y_variable_names, directories, supports=supports,
            num_workers=num_workers, allow_no_data=allow_no_data, **kwargs)

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
            supports=None, num_workers=0, allow_no_data=False, **kwargs):
        super().__init__(
            x_variable_names, y_variable_names, directories, supports=supports,
            num_workers=num_workers, allow_no_data=allow_no_data, **kwargs)

        self.data = self._load_all_data(self.data_directories)
        return

    def __getitem__(self, i):
        return self.data[i]


class SimplifiedDataset(BaseDataset):

    def __init__(
            self, x_variable_names, y_variable_names, raw_dict_x,
            prepost_converter,
            *, answer_raw_dict_y=None, num_workers=0, **kwargs):
        self.x_variable_names = x_variable_names
        self.y_variable_names = y_variable_names
        self.raw_dict_x = raw_dict_x
        self.answer_raw_dict_y = answer_raw_dict_y
        self.num_workers = num_workers
        self.prepost_converter = prepost_converter

        self.x_dict_mode = isinstance(self.x_variable_names, dict)
        self.y_dict_mode = isinstance(self.y_variable_names, dict)

        if self.x_dict_mode:
            self.first_variable_name = list(
                self.x_variable_names.values())[0][0]
        else:
            self.first_variable_name = self.x_variable_names[0]
        return

    def __len__(self):
        return 1

    def __getitem__(self, i):
        converted_dict_data = self.prepost_converter.preprocess(
            self.raw_dict_x)
        x_data = self._merge_data(
            converted_dict_data, self.x_variable_names)
        if self.answer_raw_dict_y is not None:
            converted_dict_data.update(self.prepost_converter.preprocess(
                self.answer_raw_dict_y))
            y_data = self._merge_data(
                converted_dict_data, self.y_variable_names)
        else:
            y_data = None

        return {
            'x': x_data, 't': y_data, 'data_directory': None}


class DataDict(dict):

    @property
    def device(self):
        devices = [v.device for v in self.values()]
        return devices[0]

    def to(self, device):
        return DataDict({k: v.to(device) for k, v in self.items()})


class CollateFunctionGenerator():

    def __init__(
            self, *, time_series=False, dict_input=False, dict_output=False,
            use_support=False, element_wise=False, data_parallel=False):

        if time_series:
            self.shape_length = 2
        else:
            self.shape_length = 1

        self.convert_input_dense = self._determine_convert_dense(
            time_series, dict_input, element_wise, data_parallel)

        # No need to scatter output data even when data_parallel
        self.convert_output_dense = self._determine_convert_dense(
            time_series, dict_output, element_wise)

        if dict_input:
            if data_parallel:
                self.convert_input_tensor = self._convert_dict_list
            else:
                self.convert_input_tensor = self._convert_dict_tensor
        else:
            self.convert_input_tensor = convert_tensor
        if dict_output:
            # No need to scatter output data even when data_parallel
            self.convert_output_tensor = self._convert_dict_tensor
        else:
            self.convert_output_tensor = convert_tensor

        if use_support:
            if data_parallel:
                self.convert_sparse = self._generate_sparse_list
                self.prepare_batch \
                    = self._prepare_batch_with_support_data_parallel
            else:
                self.convert_sparse = self._concatenate_sparse_sequence
                self.prepare_batch = self._prepare_batch_with_support
        else:
            self.convert_sparse = self._return_none
            self.prepare_batch = self._prepare_batch_without_support

        if element_wise:
            self.extract_original_shapes = self._return_none
        else:
            if dict_input:
                self.extract_original_shapes \
                    = self._extract_original_shapes_from_dict
            else:
                self.extract_original_shapes \
                    = self._extract_original_shapes_from_list

        return

    def _determine_convert_dense(
            self, time_series, dict_input, element_wise, data_parallel=False):
        if time_series:
            if dict_input:
                if data_parallel:
                    return self._generate_sequence_dict_list
                else:
                    return self._concatenate_sequence_dict
            else:
                if element_wise:
                    return self._stack_sequence
                else:
                    return self._pad_time_dense_sequence
        else:
            if dict_input:
                if data_parallel:
                    return self._generate_sequence_dict_list
                else:
                    return self._concatenate_sequence_dict
            else:
                if element_wise:
                    return self._stack_sequence
                else:
                    if data_parallel:
                        return self._generate_sequence_list
                    else:
                        return self._concatenate_sequence_list

    def _pad_time_dense_sequence(self, batch, key):
        data = [b[key] for b in batch]

        max_time_lengths = np.max([d.shape[0] for d in data])
        return torch.cat(
            [self._pad_time_direction(d, max_time_lengths) for d in data],
            axis=1)

    def _pad_time_direction(self, data, time_length):
        if len(data) == time_length:
            return data
        remaining_length = time_length - len(data)
        zeros = torch.zeros((remaining_length, *data.shape[1:]))
        padded_data = torch.cat([data, zeros])
        return padded_data

    def _stack_sequence(self, batch, key):
        return torch.stack([b[key] for b in batch])

    def _concatenate_sequence_list(self, batch, key):
        return torch.cat([b[key] for b in batch])

    def _concatenate_sequence_dict(self, batch, key):
        keys = batch[0][key].keys()
        return {
            dict_key:
            torch.cat([b[key][dict_key] for b in batch])
            for dict_key in keys}

    def _generate_sequence_dict_list(self, batch, key):
        keys = batch[0][key].keys()
        return {
            dict_key: [b[key][dict_key] for b in batch] for dict_key in keys}

    def _generate_sequence_list(self, batch, key):
        return [b[key] for b in batch]

    def _return_none(self, *args, **kwargs):
        return None

    def _extract_original_shapes_from_dict(self, batch):
        keys = batch[0]['x'].keys()
        return {
            dict_key:
            np.array([
                list(b['x'][dict_key].shape[:self.shape_length])
                for b in batch])
            for dict_key in keys}

    def _extract_original_shapes_from_list(self, batch):
        return np.array([b['x'].shape[:self.shape_length] for b in batch])

    def _concatenate_sparse_sequence(self, batch, key):
        sparse_infos = self._pad_sparse_sequence(batch, key)
        n_sparse_features = len(sparse_infos[0])
        return [
            merge_sparse_tensors(
                [s[i] for s in sparse_infos], return_coo=True)
            for i in range(n_sparse_features)]

    def _generate_sparse_list(self, batch, key):
        return self._pad_sparse_sequence(batch, key)

    def _pad_sparse_sequence(self, batch, key, length=None):
        return [[pad_sparse(s, length) for s in b[key]] for b in batch]

    def __call__(self, batch):
        x = self.convert_input_dense(batch, 'x')
        if batch[0]['t'] is None:
            t = torch.tensor([])
        else:
            t = self.convert_output_dense(batch, 't')
        supports = self.convert_sparse(batch, 'supports')
        original_shapes = self.extract_original_shapes(batch)
        data_directories = [b.pop('data_directory', None) for b in batch]
        return {
            'x': x, 't': t, 'supports': supports,
            'original_shapes': original_shapes,
            'data_directories': data_directories}

    def _prepare_batch_without_support(
            self, batch, device=None, output_device=None, non_blocking=False):
        return {
            'x': self.convert_input_tensor(
                batch['x'], device=device, non_blocking=non_blocking),
            'original_shapes': batch['original_shapes'],
        }, self.convert_output_tensor(
            batch['t'], device=output_device, non_blocking=non_blocking)

    def _prepare_batch_with_support(
            self, batch, device=None, output_device=None, non_blocking=False):
        return {
            'x': self.convert_input_tensor(
                batch['x'], device=device, non_blocking=non_blocking),
            'supports': [
                convert_tensor(s, device=device, non_blocking=non_blocking)
                for s in batch['supports']],
            'original_shapes': batch['original_shapes'],
        }, self.convert_output_tensor(
            batch['t'], device=output_device, non_blocking=non_blocking)

    def _prepare_batch_with_support_data_parallel(
            self, batch, device=None, output_device=None, non_blocking=False):
        return {
            'x': self.convert_input_tensor(
                batch['x'], device='cpu', non_blocking=non_blocking),
            'supports': convert_sparse_info(
                batch['supports'], device='cpu', non_blocking=non_blocking),
            'original_shapes': batch['original_shapes'],
        }, self.convert_output_tensor(
            batch['t'], device=output_device, non_blocking=non_blocking)

    def _convert_dict_tensor(self, dict_tensor, device, non_blocking):
        if len(dict_tensor) == 0:
            return convert_tensor(
                dict_tensor, device=device, non_blocking=non_blocking)
        else:
            return DataDict({
                k: convert_tensor(v, device=device, non_blocking=non_blocking)
                for k, v in dict_tensor.items()})

    def _convert_dict_list(self, dict_list, device, non_blocking):
        if len(dict_list) == 0:
            return convert_tensor(
                dict_list, device=device, non_blocking=non_blocking)
        else:
            return DataDict({k: v for k, v in dict_list.items()})


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


def convert_sparse_info(sparse_info, device=None, non_blocking=False):
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
    sparse_info: list[list[dict[str: torch.Tensor]]]
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
            converted_sparses = [
                [
                    torch.sparse_coo_tensor(
                        torch.stack([s['row'], s['col']]),
                        s['values'], s['size']
                    ).to(device)
                    for s in si]
                for si in sparse_info]
        else:
            converted_sparses = [
                torch.sparse_coo_tensor(
                    torch.stack([si['row'], si['col']]),
                    si['values'], si['size']
                ).to(device)
                for si in sparse_info]

    return converted_sparses


def merge_sparse_tensors(stripped_sparse_info, *, return_coo=True):
    """Merge sparse tensors.

    Parameters
    ----------
    stripped_sparse_info: list[dict[str: torch.Tensor]]
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

import torch
import numpy as np


# HACK: THIS IS TEMPORAL IMPLEMENTATION
# theses should be implemented in siml tensors
# HACK: split_data_if_needed, update_original_shapes functions
# should be implemented in ISimlVariables


class TensorSpliter:
    def __init__(
        self,
        input_time_series_keys: list[str],
        output_time_series_keys: list[str]
    ) -> None:
        self.input_time_series_keys = input_time_series_keys
        self.output_time_series_keys = output_time_series_keys

    def _split_data_if_needed(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        time_series_split: tuple[int, int, int]
    ):
        if time_series_split is None:
            return [x], [y]
        split_x_tensors = self._split_core(
            x['x'], self.input_time_series_keys, time_series_split)
        split_xs = [
            {
                'x': split_x_tensor,
                'original_shapes':
                self._update_original_shapes(
                    split_x_tensor, x['original_shapes']),
                'supports': x['supports']}
            for split_x_tensor in split_x_tensors]
        split_ys = self._split_core(
            y, self.output_time_series_keys, time_series_split)
        return split_xs, split_ys

    def _update_original_shapes(self, x, previous_shapes):
        if previous_shapes is None:
            return None
        if isinstance(x, torch.Tensor):
            previous_shapes[:, 0] = len(x)
            return previous_shapes
        elif isinstance(x, dict):
            return {
                k: self._update_original_shapes(v, previous_shapes[k])
                for k, v in x.items()}
        else:
            raise ValueError(f"Invalid format: {x}")

    def _split_core(self, x, time_series_keys, time_series_split):
        if isinstance(x, torch.Tensor):
            len_x = len(x)
        elif isinstance(x, dict):
            lens = np.array([
                len(v) for k, v in x.items() if k in time_series_keys])
            if not np.all(lens == lens[0]):
                raise ValueError(
                    f"Time series length mismatch: {time_series_keys}, {lens}")
            len_x = lens[0]
        else:
            raise ValueError(f"Invalid format: {x}")

        start, step, length = time_series_split
        range_ = range(start, len_x - length + 1, step)

        if isinstance(x, torch.Tensor):
            return [x[s:s+length] for s in range_]

        elif isinstance(x, dict):
            return [{
                k:
                v[s:s+length] if k in time_series_keys
                else v for k, v in x.items()} for s in range_]

        else:
            raise ValueError(f"Invalid format: {x}")

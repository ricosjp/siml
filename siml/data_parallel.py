import numpy as np
import torch

from . import datasets


class DataParallel(torch.nn.DataParallel):

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    inputs = scatter_core(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter_core(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def scatter_core(inputs, target_gpus, dim=0):

    def get_loop_variables(obj):
        if len(obj) < len(target_gpus):
            n_gpu = len(obj)
            size = 1
        else:
            n_gpu = len(target_gpus)
            size = len(obj) // len(target_gpus)
        indices = [i * size for i in range(n_gpu)]
        indices = indices + [None]
        return n_gpu, indices

    def scatter_map(obj):
        if isinstance(obj, torch.autograd.Variable):
            return torch.nn.parallel._functions.Scatter.apply(
                target_gpus, None, dim, obj)
        assert not torch.is_tensor(obj), "Tensors not supported in scatter."

        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))

        if isinstance(obj, datasets.DataDict):
            # Siml dict input
            n_gpu, indices = get_loop_variables(list(obj.values())[0])
            return [datasets.DataDict({
                key:
                torch.cat(value[indices[i]:indices[i+1]]).to(target_gpus[i])
                for key, value in obj.items()}) for i in range(n_gpu)]

        if isinstance(obj, np.ndarray):
            n_gpu, indices = get_loop_variables(obj)
            return [
                obj[indices[i]:indices[i+1], :] for i in range(n_gpu)]

        if isinstance(obj, list) and len(obj) > 0:
            # When list, simply output concatenated tensor
            n_gpu, indices = get_loop_variables(obj)
            if isinstance(obj[0], list):
                # Sparse info
                n_sparse_features = len(obj[0])
                return [
                    [
                        datasets.merge_sparse_tensors(
                            [
                                s[i_feature] for s
                                in obj[indices[i]:indices[i+1]]],
                            return_coo=True).to(target_gpus[i])
                        for i_feature in range(n_sparse_features)]
                    for i in range(n_gpu)]
            else:
                # Dense tensor
                return [
                    torch.cat(obj[indices[i]:indices[i+1]]).to(target_gpus[i])
                    for i in range(n_gpu)]

        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))

        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


import torch


def identity(x):
    return x


def max_pool(x, original_shapes):
    split_x = split(x, original_shapes)
    dim = len(original_shapes[0]) - 1
    return torch.stack([
        torch.max(
            s, dim=dim, keepdim=False
        )[0]  # [0] to unwrap tuple output of torch.max
        for s in split_x], dim=dim)


def mean(x, original_shapes):
    split_x = split(x, original_shapes)
    dim = len(original_shapes[0]) - 1
    return torch.stack([
        torch.mean(s, dim=dim, keepdim=False) for s in split_x], dim=dim)


def min(x, original_shapes):
    split_x = split(x, original_shapes)
    dim = len(original_shapes[0]) - 1
    return torch.stack([
        min_func(s, dim=dim, keepdim=False) for s in split_x], dim=dim)


def min_func(*args, **kwargs):
    ret = torch.min(*args, **kwargs)
    if isinstance(ret, tuple):
        return ret[0]
    else:
        return ret


def normalize(x):
    norms = torch.norm(x, dim=-1, keepdim=True)
    return x / (norms + 1e-5)


def split(x, original_shapes):
    if isinstance(original_shapes, dict):
        raise ValueError(
            'Input is dict. Specify dict_key in the block_setting.')
    if len(original_shapes) == 1:
        return (x,)

    if len(original_shapes[0]) == 1:
        return torch.split(x, [s[0] for s in original_shapes])
    elif len(original_shapes[0]) == 2:
        return torch.split(x, [s[1] for s in original_shapes], dim=1)
    else:
        raise ValueError(f"Unexpected original_shapes: {original_shapes}")


def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))


DICT_ACTIVATIONS = {
    'identity': identity,
    'relu': torch.relu,
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
    'max_pool': max_pool,
    'max': max_pool,
    'mean': mean,
    'mish': mish,
    'normalize': normalize,
    'softplus': torch.nn.functional.softplus,
    'sqrt': torch.sqrt,
}


import torch


def identity(x):
    return x


def max_pool(x):
    return torch.max(x, dim=-2, keepdim=True)[0]


def mean(x):
    return torch.mean(x, dim=-2, keepdim=True)


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
    'softplus': torch.nn.functional.softplus,
}

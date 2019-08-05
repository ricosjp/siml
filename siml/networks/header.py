import chainer as ch


def max_pool(x):
    return ch.functions.max(x, axis=1, keepdims=True)


DICT_ACTIVATIONS = {
    'identity': ch.functions.identity,
    'relu': ch.functions.relu,
    'sigmoid': ch.functions.sigmoid,
    'tanh': ch.functions.tanh,
    'max_pool': max_pool,
}

import chainer as ch


DICT_ACTIVATIONS = {
    'identity': ch.functions.identity,
    'relu': ch.functions.relu,
    'sigmoid': ch.functions.sigmoid,
    'tanh': ch.functions.tanh,
}

import chainer as ch

from . import adjustable_mlp
from . import gcn
from . import mlp


class Network(ch.ChainList):

    DICT_BLOCKS = {
        'mlp': mlp.MLP,
        'adjustable_mlp': adjustable_mlp.AdjustableMLP,
        'adjustable_brick_mlp': adjustable_mlp.AdjustableBrickMLP,
        'gcn': gcn.GCN,
        'res_gcn': gcn.ResGCN
    }

    def __init__(self, model_setting):
        super().__init__(*[
            self.DICT_BLOCKS[block.name](block)
            for block in model_setting.blocks])

    def __call__(self, x):
        h = x
        for link in self:
            h = link(h)
        return h

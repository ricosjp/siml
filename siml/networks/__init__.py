import pkg_resources

from .network import add_block
from .network import Network  # NOQA
from .siml_module import SimlModule  # NOQA
from . import activation
from . import concatenator
from . import array2diagmat
from . import array2symmat
from . import deepsets
from . import gcn
from . import grad_gcn
from . import identity
from . import integration
from . import iso_gcn
from . import laplace_net
from . import lstm
from . import mlp
from . import message_passing
from . import reducer
from . import reshape
from . import symmat2array
from . import tcn
from . import tensor_operations
from . import time_norm
from . import translator


# Add block information
blocks = [
    # Layers without weights
    activation.Activation,
    array2diagmat.Array2Diagmat,
    array2symmat.Array2Symmat,
    concatenator.Concatenator,
    tensor_operations.Contraction,
    identity.Identity,
    integration.Integration,
    reducer.Reducer,
    reshape.Reshape,
    symmat2array.Symmat2Array,
    time_norm.TimeNorm,
    translator.Translator,

    # Layers with weights
    deepsets.DeepSets,
    gcn.GCN,
    grad_gcn.GradGCN,
    iso_gcn.IsoGCN,
    laplace_net.LaplaceNet,
    lstm.LSTM,
    mlp.MLP,
    message_passing.MessagePassing,
    tcn.TCN,
]

# # Layers without weights
# 'activation': activation.Activation,
# 'array2diagmat': array2diagmat.Array2Diagmat,
# 'array2symmat': array2symmat.Array2Symmat,
# 'contraction': tensor_operations.Contraction,
# 'integration': integration.Integration,
# 'reducer': reducer.Reducer,
# 'reshape': reshape.Reshape,
# 'symmat2array': symmat2array.Symmat2Array,
# 'time_norm': time_norm.TimeNorm,
#
# # Layers with weights
# 'deepsets': deepsets.DeepSets,
# 'gcn': gcn.GCN,
# 'grad_gcn': grad_gcn.GradGCN,
# 'iso_gcn': iso_gcn.IsoGCN,
# 'laplace_net': laplace_net.LaplaceNet,
# 'lstm': lstm.LSTM,
# 'message_passing': message_passing.MessagePassing,
# 'tcn': tcn.TCN,

for block in blocks:
    add_block(block)

# Add aliases for backward compatibility
Network.dict_block_class.update({
    'distributor': reducer.Reducer,
    'adjustable_mlp': mlp.MLP,
})


# Load torch_geometric blocks when torch_geometric is installed
if 'torch-geometric' in [
        p.key for p in pkg_resources.working_set]:  # pylint: disable=E1133
    from . import pyg
    pyg_blocks = [
        pyg.ClusterGCN,
        pyg.GIN,
        pyg.GCNII,
    ]
    for py_block in pyg_blocks:
        add_block(py_block)

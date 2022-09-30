import pkg_resources

from .network import add_block
from .network import Network  # NOQA
from .siml_module import SimlModule  # NOQA
from . import activation
from . import concatenator
from . import array2diagmat
from . import array2symmat
from . import boundary
from . import deepsets
from . import gcn
from . import group
from . import id_mlp
from . import identity
from . import integration
from . import iso_gcn
from . import lstm
from . import mlp
from . import message_passing
from . import normalized_mlp
from . import penn
from . import pinv_mlp
from . import proportional
from . import reducer
from . import reshape
from . import set_transformer
from . import share
from . import spmm
from . import symmat2array
from . import tcn
from . import tensor_operations
from . import time_norm
from . import translator
from . import projection
from . import threshold


# Add block information
blocks = [
    # Layers without weights
    activation.Activation,
    array2diagmat.Array2Diagmat,
    array2symmat.Array2Symmat,
    boundary.Assignment,
    boundary.Dirichlet,
    boundary.Interaction,
    boundary.NeumannIsoGCN,
    boundary.NeumannEncoder,
    boundary.NeumannDecoder,
    concatenator.Concatenator,
    tensor_operations.Contraction,
    identity.Identity,
    integration.Integration,
    reducer.Reducer,
    reshape.Reshape,
    reshape.TimeSeriesToFeatures,
    reshape.FeaturesToTimeSeries,
    reshape.Accessor,
    spmm.SpMM,
    symmat2array.Symmat2Array,
    tensor_operations.TensorProduct,
    time_norm.TimeNorm,
    translator.Translator,
    projection.Projection,
    threshold.Threshold,

    # Layers with weights
    deepsets.DeepSets,
    gcn.GCN,
    group.Group,
    id_mlp.IdMLP,
    iso_gcn.IsoGCN,
    lstm.LSTM,
    mlp.MLP,
    message_passing.MessagePassing,
    normalized_mlp.NormalizedMLP,
    penn.PENN,
    pinv_mlp.PInvMLP,
    proportional.Proportional,
    set_transformer.SetTransformerEncoder,
    set_transformer.SetTransformerDecoder,
    share.Share,
    tcn.TCN,
    tensor_operations.EquivariantMLP,
]

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

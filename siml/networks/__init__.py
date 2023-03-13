import pkg_resources

from .network import add_block
from .network import Network  # NOQA
from .siml_module import SimlModule  # NOQA
from . import activation
from . import array2diagmat
from . import array2symmat
from . import boundary
from . import concatenator
from . import deepsets
from . import einsum
from . import gcn
from . import group
from . import id_mlp
from . import identity
from . import integration
from . import iso_gcn
from . import lstm
from . import message_passing
from . import mlp
from . import nan_mlp
from . import normalized_mlp
from . import penn
from . import pinv_mlp
from . import projection
from . import proportional
from . import reducer
from . import reshape
from . import set_transformer
from . import share
from . import spmm
from . import symmat2array
from . import tcn
from . import tensor_operations
from . import threshold
from . import time_norm
from . import translator
from . import upper_limit


# Add block information
blocks = [
    # Layers without weights
    activation.Activation,
    array2diagmat.Array2Diagmat,
    array2symmat.Array2Symmat,
    boundary.Assignment,
    boundary.Dirichlet,
    boundary.Interaction,
    boundary.NeumannDecoder,
    boundary.NeumannEncoder,
    boundary.NeumannIsoGCN,
    concatenator.Concatenator,
    einsum.EinSum,
    identity.Identity,
    integration.Integration,
    projection.Projection,
    reducer.Reducer,
    reshape.Accessor,
    reshape.FeaturesToTimeSeries,
    reshape.Reshape,
    reshape.TimeSeriesToFeatures,
    spmm.SpMM,
    symmat2array.Symmat2Array,
    tensor_operations.Contraction,
    tensor_operations.TensorProduct,
    threshold.Threshold,
    time_norm.TimeNorm,
    translator.Translator,
    upper_limit.UpperLimit,

    # Layers with weights
    deepsets.DeepSets,
    gcn.GCN,
    group.Group,
    id_mlp.IdMLP,
    iso_gcn.IsoGCN,
    lstm.LSTM,
    message_passing.MessagePassing,
    mlp.MLP,
    nan_mlp.NaNMLP,
    normalized_mlp.NormalizedMLP,
    penn.PENN,
    pinv_mlp.PInvMLP,
    proportional.Proportional,
    set_transformer.SetTransformerDecoder,
    set_transformer.SetTransformerEncoder,
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

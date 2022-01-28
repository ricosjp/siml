import torch


def mul(sparse, tensor):
    """Multiply sparse tensors and tensors.

    Parameters
    ----------
    sparses: torch.sparse_coo_tensor
    tensor: torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    shape = tensor.shape
    sparse_shape = sparse.shape
    h = torch.reshape(
        sparse.mm(torch.reshape(tensor, (shape[0], -1))),
        [sparse_shape[0]] + list(shape[1:]))
    return h

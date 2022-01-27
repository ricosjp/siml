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
    tensor_rank = len(shape) - 2
    if tensor_rank == 0:
        h = sparse.mm(tensor)
    elif tensor_rank > 0:
        dim = tensor.shape[-2]
        h = torch.stack([
            mul(sparse, tensor[:, i_dim])
            for i_dim in range(dim)], dim=1)
    else:
        raise ValueError(f"Tensor shape invalid: {shape}")

    return h

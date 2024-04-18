"""Implicit GNN

Original Implementation is found here: https://github.com/SwiftieH/IGNN


* Overview
  * Fangda Gu, Heng Chang, Wenwu Zhu, Somayeh Sojoudi, Laurent El Ghaoui
  * 34th Conference on Neural Information Processing Systems (NeurIPS 2020), Vancouver, Canada.
  * arxiv: https://arxiv.org/abs/2009.06211

"""

from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.sparse
from torch import nn
from torch.autograd import Function
from torch.nn import Parameter

from siml.setting import BlockSetting

from . import siml_module


def calculate_spectral_rad(sparse_tensor: torch.Tensor) -> np.ndarray:
    """Compute spectral radius from a tensor

    Parameters
    ----------
    sparse_tensor : torch.Tensor
        tensor to calculate

    Returns
    -------
    np.ndarray
        spectral radius
    """
    A = sparse_tensor.data.coalesce().cpu()
    A_scipy = sp.coo_matrix(
        (np.abs(A.values().numpy()), A.indices().numpy()), shape=A.shape
    )
    return np.abs(sp.linalg.eigs(A_scipy, k=1, return_eigenvectors=False)[0])


class ImplicitGNN(siml_module.SimlModule):
    """
    A Implicit Graph Neural Network Layer (IGNN)
    """

    @staticmethod
    def get_name():
        return "implicit_gnn"

    @staticmethod
    def is_trainable():
        return True

    @staticmethod
    def accepts_multiple_inputs():
        return False

    @staticmethod
    def uses_support():
        return True

    def __init__(self, block_setting: BlockSetting):
        super().__init__(
            block_setting,
            create_linears=False,
            create_activations=True,
            create_dropouts=False,
        )

        self.kappa = block_setting.optional.get("kappa", 0.99)

        self._validate_setting()

        nodes = self.block_setting.nodes
        self.weight = Parameter(torch.FloatTensor(nodes[0], nodes[1]))
        nn.init.xavier_normal_(self.weight)

        # assumed feature dimension of U is same as that of X
        self.omega = Parameter(torch.FloatTensor(nodes[0], nodes[1]))
        nn.init.xavier_normal_(self.omega)

    def _validate_setting(self):
        nodes = self.block_setting.nodes
        if len(nodes) != 2:
            raise ValueError(f"size of nodes must be two. input: {nodes}")

        if len(self.activations) > 1:
            raise ValueError(
                "the num. of activation functions must be one. "
                f"input: {self.activations}"
            )

    def forward(
        self,
        x: torch.Tensor,
        supports: Optional[list[torch.Tensor]],
        *args,
        **kwards,
    ):
        if len(x.shape) != 2:
            raise NotImplementedError(
                "For now, only zero rank tensor is allowed."
            )

        assert len(supports) == 1
        out = self._forward(
            X_0=x.T, A=supports[0], U=x.T, phi=self.activations[0]
        )
        return out.T

    def _forward(
        self,
        X_0: torch.Tensor,
        A: torch.Tensor,
        U: torch.Tensor,
        phi: Callable[[torch.Tensor], torch.Tensor],
        fw_mitr: int = 300,
        bw_mitr: int = 300,
        A_orig: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward Process of Implicit Graph

        Parameters
        ----------
        X_0 : torch.Tensor
            tensor of initial value (shape : [n_node, n_feature])
        A : torch.Tensor
            adjacent matrix tensor
        U : torch.Tensor
            feature matrix provided as additional information of each node
            (shape : [n_node, n_feature])
        phi : Callable[[torch.Tensor], torch.Tensor]
            activation function
        fw_mitr : int, optional
            the maximum number of forward iterations, by default 300
        bw_mitr : int, optional
            the maximum number of backward iterations, by default 300
        A_orig : Optional[torch.Tensor], optional
            If fed, A matrix is replaced with this matrix, by default None

        """
        if self.kappa is not None:  # when self.k = 0, A_rho is not required
            A_rho = calculate_spectral_rad(A)
            tol: float = 1e-5
            projection_norm_inf(
                self.weight, criteria=self.kappa / (A_rho + tol)
            )

        # b_omega = omega U A
        # In order to account for feature information from neighbouring nodes
        b_omega = torch.spmm(torch.transpose(U, 0, 1), self.omega.T).T
        b_omega = torch.spmm(torch.transpose(A, 0, 1), b_omega.T).T

        return ImplicitFunction.apply(
            self.weight,
            X_0,
            A if A_orig is None else A_orig,
            b_omega,
            phi,
            fw_mitr,
            bw_mitr,
        )


class IGNNIterationStatus(Enum):
    not_started = 0
    converged = 1
    reached_max_itration = 2


class ImplicitFunction(Function):
    # ImplicitFunction.apply(input, A, U, self.X_0, self.W, self.Omega_1, self.Omega_2)
    @staticmethod
    def forward(
        ctx: Any,
        W: torch.Tensor,
        X_0: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        phi: Callable[[torch.Tensor], torch.Tensor],
        fd_mitr: int = 300,
        bw_mitr: int = 300,
    ):
        X_0 = B if X_0 is None else X_0
        X, err, status, D = ImplicitFunction._forward_iteration(
            W, X_0, A, B, phi, n_itr=fd_mitr
        )
        # X, err, status, D = ImplicitFunction.inn_pred(W, X_0, A, B, phi, mitr=fd_mitr, compute_dphi=True)
        ctx.save_for_backward(W, X, A, B, D, X_0, torch.tensor(bw_mitr))
        if status != IGNNIterationStatus.converged:
            print(
                f"Iterations not converging! err: {err}, status: {status.name}"
            )
        return X

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, *grad_outputs):
        W, X, A, B, D, X_0, bw_mitr = ctx.saved_tensors

        bw_mitr = bw_mitr.detach().numpy()
        grad_x = grad_outputs[0]

        def dphi(X): return torch.mul(X, D)
        grad_z, err, status = ImplicitFunction._backward_iteration(
            W, X_0, A, grad_x, dphi, n_itr=bw_mitr
        )

        grad_W = grad_z @ torch.spmm(A, X.T)
        grad_B = grad_z

        # Might return gradient for A if needed
        # NOTE: According to Pytorch Official Documment, We return variables as many as that in grad_outputs.
        # Gradients of non-Tensor arguments must be None.
        return grad_W, None, torch.zeros_like(A), grad_B, None, None, None

    @staticmethod
    def _forward_iteration(
        W: torch.Tensor,
        X: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        phi: Callable[[torch.Tensor], torch.Tensor],
        n_itr: int = 300,
        tol: float = 3e-6,
        compute_dphi: bool = True,
    ):
        status = IGNNIterationStatus.reached_max_itration
        for _ in range(n_itr):
            support = torch.spmm(A, (W @ X).T).T
            X_new = phi(support + B)
            # err = torch.norm(X_new - X, np.inf)  # it is deprecated
            # https://pytorch.org/docs/stable/generated/torch.linalg.matrix_norm.html#torch.linalg.matrix_norm
            # The second argument is type of norm. inf norm
            err = torch.linalg.matrix_norm(X_new - X, np.inf)
            if err < tol:
                status = IGNNIterationStatus.converged
                break
            X = X_new

        if not compute_dphi:
            return X_new, err, status, None

        D: torch.Tensor
        with torch.enable_grad():
            support = torch.spmm(A, (W @ X).T).T
            Z = support + B
            # requires_grad_()’s main use case is to tell autograd to begin recording operations on a Tensor tensor.
            Z.requires_grad_(True)
            X_new = phi(Z)
            # element-wise derivative
            # D is the same as Z
            D = torch.autograd.grad(torch.sum(X_new), Z, only_inputs=True)[0]

        return X_new, err, status, D

    @staticmethod
    def _backward_iteration(
        W: torch.Tensor,
        grad_Z: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        dphi: Callable[[torch.Tensor], torch.Tensor],
        n_itr: int = 300,
        tol: float = 3e-6,
    ):
        status = IGNNIterationStatus.reached_max_itration
        At = torch.transpose(A, 0, 1)
        for _ in range(n_itr):
            support = torch.spmm(At, (W.T @ grad_Z).T).T
            grad_Z_new = dphi(support + B)
            # https://pytorch.org/docs/stable/generated/torch.linalg.matrix_norm.html#torch.linalg.matrix_norm
            # The second argument is type of norm. inf norm
            err = torch.linalg.matrix_norm(grad_Z_new - grad_Z, np.inf)
            # err = torch.norm(grad_Z_new - grad_Z, np.inf)  # it is deprecated
            if err < tol:
                status = IGNNIterationStatus.converged
                break
            grad_Z = grad_Z_new

        return grad_Z_new, err, status


def projection_norm_inf(A: torch.Tensor, criteria: float = 0.99) -> None:
    """Project onto ||A||_inf <= criteria return updated A

    Parameters
    ----------
    A : torch.Tensor
        Tensor object
    criteria : float, optional
        criteria value, by default 0.99
    """
    A_np = A.clone().detach().cpu().numpy()
    inf_norms = np.abs(A_np).sum(axis=-1)

    positions = np.where(inf_norms > criteria)
    A_np[positions] *= (criteria / inf_norms[positions])[:, np.newaxis]

    # vertify
    epsilon = 1e-5
    assert np.linalg.norm(A_np, ord=np.inf) <= criteria + epsilon

    A.data.copy_(
        torch.tensor(A_np, dtype=A.dtype, device=A.device), non_blocking=True
    )
    return

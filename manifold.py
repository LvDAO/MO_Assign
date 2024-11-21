"""
This Python file defines two manifold computation methods:
1. Contraction mapping based on Christoffel symbols and local coordinates.
2. Contraction mapping based on SVD and other methods.
To make function objects easy to obtain derivatives, we use torch to implement most functionality.
"""

import torch
import torch.nn as nn
import torchdiffeq
from torch import vmap, einsum
from torch.func import jacrev
from functools import wraps

import numpy as np


def flatten_if_needed(func):
    """Decorator: Flattens high dimensional tensors if total elements equals dim"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        flattened_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                if arg.dim() > 1 and arg.numel() == self.dim:
                    flattened_args.append(arg.reshape(-1))
                else:
                    flattened_args.append(arg)
            else:
                flattened_args.append(arg)
        return func(self, *flattened_args, **kwargs)

    return wrapper


class Metric(nn.Module):
    """
    This class defines the metric on a manifold, using Euclidean metric by default.
    To define a new metric, override the forward function.

    Args:
        dim (int): Dimension of the manifold
    """

    def __init__(self, dim):
        super(Metric, self).__init__()
        self.dim = dim
        self.forward_v = vmap(
            vmap(self.forward, in_dims=(None, None, 0)), in_dims=(None, 0, None)
        )

    @flatten_if_needed
    def get_matrix(self, x):
        eye_mat = torch.eye(self.dim, device=x.device)
        return self.forward_v(x, eye_mat, eye_mat)

    def get_inverse(self, x):
        return self.get_matrix(x).inverse()

    @flatten_if_needed
    def forward(self, x, v1, v2):
        if (
            not isinstance(x, torch.Tensor)
            or not isinstance(v1, torch.Tensor)
            or not isinstance(v2, torch.Tensor)
        ):
            raise TypeError("x, v1 and v2 must be torch.Tensor type")

        # Ensure correct dimensions
        if x.size(0) != self.dim or v1.size(0) != self.dim or v2.size(0) != self.dim:
            raise ValueError(f"Input dimensions must be {self.dim}")

        return torch.sum(v1 * v2)  # Euclidean metric


class PoincareDiskMetric(Metric):
    """Riemannian metric for the Poincare disk model"""

    @flatten_if_needed
    def forward(self, x, v1, v2):
        if (
            not isinstance(x, torch.Tensor)
            or not isinstance(v1, torch.Tensor)
            or not isinstance(v2, torch.Tensor)
        ):
            raise TypeError("x, v1 and v2 must be torch.Tensor type")

        if x.dim() != 1 or v1.dim() != 1 or v2.dim() != 1:
            raise ValueError("x, v1 and v2 must be 1-dimensional tensors")

        if x.size(0) != self.dim or v1.size(0) != self.dim or v2.size(0) != self.dim:
            raise ValueError(f"x, v1 and v2 must have dimension {self.dim}")

        norm_x_squared = torch.sum(x * x)
        scaling_factor = 4.0 / (1.0 - norm_x_squared) ** 2
        return scaling_factor * torch.sum(v1 * v2)


class Manifold:
    def __init__(self, dim, metric: Metric = None):
        self.dim = dim
        self.metric = metric if metric is not None else Metric(dim)
        self.gamma = self.christoffel_symbols(self.metric)

    @staticmethod
    def christoffel_symbols(metric: Metric):
        class ChristoffelSymbols(nn.Module):
            def __init__(self, metric: Metric):
                super(ChristoffelSymbols, self).__init__()
                self.metric = metric

            def forward(self, x):
                gkr = self.metric.get_inverse(x)
                gij = self.metric.get_matrix
                dgijdxr = jacrev(gij, argnums=0)
                dgijdxr = dgijdxr(x)
                return (
                    -0.5 * torch.einsum("kr,ijr->kij", gkr, dgijdxr)
                    + 0.5 * torch.einsum("kr,irj->kij", gkr, dgijdxr)
                    + 0.5 * torch.einsum("kr,jri->kij", gkr, dgijdxr)
                )

        return ChristoffelSymbols(metric)

    @flatten_if_needed
    def exponential_map(
        self,
        x,
        v,
        geodesics=False,
        time_steps=100,
        return_whole=False,
        return_velocity=False,
    ):

        if not geodesics:
            t = torch.tensor([0, 1], dtype=torch.float32, device=x.device)
        else:
            t = torch.linspace(0, 1, time_steps, dtype=torch.float32, device=x.device)
        velocity = lambda t, phase: torch.vstack(
            [
                phase[1],
                torch.einsum("kij,i,j->k", self.gamma(phase[0]), phase[1], phase[1]),
            ]
        )
        initial_phase = torch.vstack([x, v])
        if return_whole:
            if return_velocity:
                return torchdiffeq.odeint(
                    velocity,
                    initial_phase,
                    t,
                    options={"dtype": torch.float32},
                )
            else:
                return torchdiffeq.odeint(
                    velocity,
                    initial_phase,
                    t,
                    options={"dtype": torch.float32},
                )[:, 0]
        else:
            if return_velocity:
                return torchdiffeq.odeint(
                    velocity,
                    initial_phase,
                    t,
                    options={"dtype": torch.float32},
                )[-1]
            else:
                return torchdiffeq.odeint(
                    velocity,
                    initial_phase,
                    t,
                    options={"dtype": torch.float32},
                )[-1, 0]


class StiefelManifold(Manifold):
    def __init__(self, rdim, kdim):

        super(StiefelManifold, self).__init__(
            rdim * kdim,
        )
        self.rdim = rdim
        self.kdim = kdim
        self.inner_dim = rdim * kdim - (kdim * (kdim + 1) // 2)

    def project(self, x, v):
        """
        Project a vector onto the tangent space of the Stiefel manifold

        Args:
            x: Point on the manifold with shape (rdim, kdim)
            v: Vector to be projected with shape (rdim, kdim)

        Returns:
            Projected vector with the same shape as input
        """
        # Reshape dimensions
        x = x.reshape(self.rdim, self.kdim)
        v = v.reshape(self.rdim, self.kdim)

        # Calculate projection
        xTv = x.T @ v
        vTx = xTv.T
        proj = v - x @ ((xTv + vTx) / 2)

        return proj

    def exponential_map(self, x, v):
        state = torch.stack([x, v])

        def velocity(t, input):
            dxdt = input[1]
            dvdt = -input[0] @ input[1].T @ input[1]
            return torch.stack([dxdt, dvdt])

        return torchdiffeq.odeint(
            velocity, state, torch.tensor([0, 1], dtype=torch.float64, device=x.device)
        )[-1][0]

    def svd_retract(self, x, v):
        u, s, vh = torch.svd(x + v)
        return u @ vh.T

    def qr_retract(self, x, v, t=1):
        Y = x + t * v
        Q = torch.zeros_like(Y, device=x.device)
        R = torch.zeros((self.kdim, self.kdim), device=x.device)
        # Initialize first column
        v = Y[:, 0].clone()
        R[0, 0] = torch.norm(v)
        Q[:, 0] = v / R[0, 0]

        # Optimize subsequent column calculations using matrix operations
        for i in range(1, self.kdim):
            v = Y[:, i].clone()
            # Calculate projection coefficients for first i columns at once
            R[:i, i] = Q[:, :i].T @ v
            # Subtract all projection components from v
            v = v - Q[:, :i] @ R[:i, i]

            # Calculate norm of i-th column and normalize
            R[i, i] = torch.norm(v)
            if R[i, i] > 1e-10:
                Q[:, i] = v / R[i, i]
            else:
                Q[:, i] = torch.zeros_like(v)

        return Q

    def cayley_retract(self, x, v, t=1):
        WX = torch.eye(self.rdim, device=x.device) - x @ x.T
        U = torch.cat([WX @ v, x], dim=1)
        Z = torch.cat([x, -WX @ v], dim=1)
        RxV = (
            x
            + t
            * U
            @ (torch.eye(2 * self.kdim, device=x.device) - t * Z.T @ U / 2).inverse()
            @ Z.T
            @ x
        )
        return RxV

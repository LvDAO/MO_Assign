"""
This file contains the code for the CGM method and Newton's method for optimization on manifolds.
"""

import torch
import torch.nn as nn
from manifold import *
from torch.func import jacrev
import matplotlib.pyplot as plt
from typing import Dict, List, Any


def compute_direction(
    point,
    manifold: StiefelManifold,
    function: nn.Module,
    kappa: float,
    iter: int,
    sigma_k: torch.Tensor,
):
    # Check if kappa is in [0,1]
    if kappa <= 0 or kappa >= 1:
        raise ValueError("kappa must be in (0,1)")
    gradient = manifold.project(point, jacrev(function)(point)).flatten()
    gradient_norm = torch.norm(gradient, p=2)
    epsilon = (
        torch.min(
            torch.tensor(0.5, device=point.device), gradient_norm.clone().detach()
        )
        * gradient_norm
    )
    B_k = (
        jacrev(jacrev(function))(point).reshape(
            point.shape[0] * point.shape[1], point.shape[0] * point.shape[1]
        )
        + torch.eye(point.shape[0] * point.shape[1], device=point.device) * sigma_k
    )
    z = torch.zeros_like(point).flatten()
    r = -gradient.flatten()
    p = r

    for i in range(iter):
        pBp = p @ B_k @ p
        if pBp <= 0:
            if i == 0:
                return r.reshape(point.shape)
            else:
                return z.reshape(point.shape)
        alpha = (r @ r) / pBp
        z_new = z + alpha * p
        r_new = (
            r
            - alpha * manifold.project(point, (B_k @ p).reshape(point.shape)).flatten()
        )
        if -z_new @ gradient < kappa * gradient_norm * torch.norm(z_new, p=2):
            return z.reshape(point.shape)
        if torch.norm(r_new, p=2) < epsilon:
            return z_new.reshape(point.shape)
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        z = z_new
        r = r_new
    return z.reshape(point.shape)


def bp_length(manifold, function, c_1, gamma, point, direction, t_init=1):
    tk = t_init
    while function(manifold.qr_retract(point, direction, t=tk)) > (
        function(point)
        + c_1 * tk * (direction.flatten() @ jacrev(function)(point).flatten())
    ):
        tk *= gamma
    return tk


class ManifoldOptimizer:
    def __init__(self, manifold: Manifold):
        self.manifold = manifold
        self.log_data: Dict[str, List[Any]] = {
            "iteration": [],
            "function_value": [],
            "gradient_norm": [],
            "step_size": [],
        }

    def optimize(self, start_point, function, max_iter):
        raise NotImplementedError

    def _print_progress(self, iter_num, function_val, gradient_norm, step_size):
        print(
            f"iter: {iter_num}, function: {function_val:.20f}, "
            f"gradient_norm: {gradient_norm:.20f}, "
            f"gradient_norm_length: {step_size:.20f}"
        )
        # Record logs
        self.log_data["iteration"].append(iter_num)
        self.log_data["function_value"].append(function_val)
        self.log_data["gradient_norm"].append(gradient_norm)
        self.log_data["step_size"].append(step_size)

    def get_and_clear_log(self) -> Dict[str, List[Any]]:
        """Get and clear logs"""
        log_copy = self.log_data.copy()
        self.log_data = {
            "iteration": [],
            "function_value": [],
            "gradient_norm": [],
            "step_size": [],
        }
        return log_copy


class GradientDescent(ManifoldOptimizer):
    def __init__(self, manifold: StiefelManifold, c_1=0.25, epsilon=1e-6):
        super().__init__(manifold)
        self.c_1 = c_1
        self.epsilon = epsilon

    def optimize(self, start_point, function, max_iter):
        point = start_point
        iter_num = 0
        function_values = [function(point).item()]

        while iter_num < max_iter:
            direction = -self.manifold.project(point, jacrev(function)(point))
            tk = bp_length(self.manifold, function, self.c_1, 0.9, point, direction)
            point_new = self.manifold.qr_retract(point, direction, t=tk)
            point = point_new
            iter_num += 1
            function_values.append(function(point).item())

            self._print_progress(
                iter_num,
                function(point).item(),
                torch.norm(direction.flatten(), p=2).item(),
                tk * torch.norm(direction.flatten(), p=2).item(),
            )

            if torch.norm(direction.flatten(), p=2) < self.epsilon:
                break

        return point, function_values


class BBMethod(ManifoldOptimizer):
    """
    Barzilai-Borwein method for optimization on manifolds.
    This method adaptively chooses step sizes using gradient information.
    """

    def __init__(
        self,
        manifold: StiefelManifold,
        alpha=1,  # Initial step size
        alpha_max=2,  # Maximum step size
        alpha_min=0.5,  # Minimum step size
        rho=0.9,  # Line search parameter
        c_1=0.25,  # Armijo condition parameter
        epsilon=1e-6,  # Convergence tolerance
        M: int = 10,  # Memory parameter
    ):
        super().__init__(manifold)
        self.alpha = alpha
        self.c_1 = c_1
        self.epsilon = epsilon
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.M = M
        self.rho = rho
        self._validate_params()

    def _validate_params(self):
        if self.alpha_max <= self.alpha_min:
            raise ValueError("alpha_max must be greater than alpha_min")
        if not isinstance(self.M, int) or self.M <= 0:
            raise ValueError("M must be a positive integer")
        if self.epsilon <= 0 or self.epsilon >= 1:
            raise ValueError("epsilon must be in (0,1)")
        if self.rho <= 0 or self.rho >= 1:
            raise ValueError("rho must be in (0,1)")

    def optimize(self, start_point, function, max_iter):
        """
        Optimize using BB method with alternating BB step sizes
        """
        point = start_point
        iter_num = 0
        function_values = [function(point).item()]

        while iter_num < max_iter:
            gradient = self.manifold.project(point, jacrev(function)(point))
            if torch.norm(gradient.flatten(), p=2) < self.epsilon:
                break

            # Compute BB step size
            if iter_num == 0:
                alpha = self.alpha
            else:
                s_km1 = (point - point_m1).flatten()  # Step difference
                y_km1 = (gradient - gradient_m1).flatten()  # Gradient difference

                # BB1 (long step) and BB2 (short step) formulas
                alpha_lbb = s_km1 @ s_km1 / torch.abs(s_km1 @ y_km1)
                alpha_sbb = torch.abs(s_km1 @ y_km1) / (y_km1 @ y_km1)

                # Alternate between BB1 and BB2
                alpha_abb = alpha_lbb if iter_num % 2 == 0 else alpha_sbb
                alpha = torch.min(
                    torch.tensor(self.alpha_max),
                    torch.max(torch.tensor(self.alpha_min), alpha_abb),
                )

            # Line search to ensure sufficient decrease
            alpha = bp_length(
                self.manifold,
                function,
                self.c_1,
                self.rho,
                point,
                -gradient,
                alpha,
            )

            point_new = self.manifold.qr_retract(point, -gradient, t=alpha)
            gradient_m1 = gradient
            point_m1 = point
            point = point_new
            iter_num += 1
            function_values.append(function(point).item())

            self._print_progress(
                iter_num,
                function(point).item(),
                torch.norm(gradient.flatten(), p=2).item(),
                alpha,
            )

        return point, function_values


class RegularizedNewton(ManifoldOptimizer):
    def __init__(
        self,
        manifold: StiefelManifold,
        sigma_0=0.1,
        c_1=0.25,
        gamma=0.9,
        eta_low=0.3,
        eta_up=0.7,
        gamma_low=0.9,
        gamma_up=1.1,
    ):
        super().__init__(manifold)
        self.sigma_0 = sigma_0
        self.c_1 = c_1
        self.gamma = gamma
        self.eta_low = eta_low
        self.eta_up = eta_up
        self.gamma_low = gamma_low
        self.gamma_up = gamma_up

        self._validate_params()

    def _validate_params(self):
        if self.sigma_0 <= 0:
            raise ValueError("sigma_0 must be positive")
        if self.c_1 <= 0 or self.c_1 >= 0.5:
            raise ValueError("c_1 must be in (0,0.5)")
        if self.gamma <= 0 or self.gamma >= 1:
            raise ValueError("gamma must be in (0,1)")
        if self.eta_low <= 0 or self.eta_low >= 1:
            raise ValueError("eta_low must be in (0,1)")
        if self.eta_up <= self.eta_low:
            raise ValueError("eta_up must be greater than eta_low")
        if self.gamma_low <= 0 or self.gamma_low >= 1:
            raise ValueError("gamma_low must be in (0,1)")
        if self.gamma_up <= 1:
            raise ValueError("gamma_up must be greater than 1")
        if self.gamma_up <= self.gamma_low:
            raise ValueError("gamma_up must be greater than gamma_low")

    def optimize(self, start_point, function, max_iter):
        point = start_point
        sigma_k = self.sigma_0
        iter_num = 0
        function_values = [function(point).item()]

        while iter_num < max_iter:
            direction = compute_direction(
                point, self.manifold, function, 0.5, 20, sigma_k
            )
            if torch.isnan(direction).any():
                print("direction is nan")
                break

            tk = bp_length(
                self.manifold, function, self.c_1, self.gamma, point, direction
            )
            point_new = self.manifold.qr_retract(point, direction, t=tk)

            rho_k_up = function(point_new) - function(point)
            rho_k_down = (
                (tk * direction.flatten()) @ jacrev(function)(point).flatten()
                + 0.5
                * (
                    (tk * direction.flatten())
                    @ jacrev(jacrev(function))(point).reshape(
                        point.shape[0] * point.shape[1], point.shape[0] * point.shape[1]
                    )
                    @ (tk * direction.flatten())
                )
                + 0.5
                * sigma_k
                * (tk * direction.flatten())
                @ (tk * direction.flatten())
            )
            rho = rho_k_up / rho_k_down

            self._print_progress(
                iter_num,
                function(point).item(),
                torch.norm(direction.flatten(), p=2).item(),
                tk * torch.norm(direction.flatten(), p=2).item(),
            )

            if rho < self.eta_low:
                sigma_k = self.gamma_up * sigma_k
            elif rho > self.eta_up:
                sigma_k = self.gamma_low * sigma_k

            point = point_new if rho >= self.eta_low else point
            function_values.append(function(point).item())
            iter_num += 1

        return point, function_values

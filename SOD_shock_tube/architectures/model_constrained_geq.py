import torch
import torch.nn as nn

from .model import DenseNet


def _flatten_macro_state(u0):
    return u0.permute(0, 2, 3, 1).reshape(u0.size(0) * u0.size(2) * u0.size(3), 4)


def _flatten_population(population):
    if population.ndim == 4:
        return population.permute(0, 2, 3, 1).reshape(-1, population.shape[1])
    if population.ndim == 3:
        return population.permute(1, 2, 0).reshape(-1, population.shape[0])
    raise ValueError(f"Unsupported population shape: {tuple(population.shape)}")


def _build_moment_matrix(basis):
    ones = torch.ones_like(basis[:, 0])
    return torch.stack([ones, basis[:, 0], basis[:, 1]], dim=0)


def _project_to_nonconserved(logits, moment_matrix):
    gram = moment_matrix @ moment_matrix.transpose(0, 1)
    projector = torch.eye(
        moment_matrix.shape[1],
        device=moment_matrix.device,
        dtype=moment_matrix.dtype,
    ) - moment_matrix.transpose(0, 1) @ torch.linalg.inv(gram) @ moment_matrix
    return logits @ projector.transpose(0, 1)


def _geq_targets(flat_macro_state, cv):
    rho = flat_macro_state[:, 0]
    ux = flat_macro_state[:, 1]
    uy = flat_macro_state[:, 2]
    temperature = flat_macro_state[:, 3]
    kinetic = ux.square() + uy.square()
    energy = cv * temperature + 0.5 * kinetic
    enthalpy = energy + temperature
    return torch.stack(
        [
            2.0 * rho * energy,
            2.0 * rho * ux * enthalpy,
            2.0 * rho * uy * enthalpy,
        ],
        dim=-1,
    )


def _solve_constrained_geq(log_base, basis, target_moments, max_iters, tolerance):
    moment_matrix = _build_moment_matrix(basis).to(device=log_base.device, dtype=log_base.dtype)
    identity = torch.eye(
        moment_matrix.shape[0], device=log_base.device, dtype=log_base.dtype
    ).unsqueeze(0)
    multipliers = torch.zeros(
        (target_moments.shape[0], target_moments.shape[1]),
        device=log_base.device,
        dtype=log_base.dtype,
    )

    def evaluate(candidate):
        exponent = (log_base + candidate @ moment_matrix).clamp(min=-60.0, max=60.0)
        population = torch.exp(exponent)
        moments = population @ moment_matrix.transpose(0, 1)
        residual = moments - target_moments
        return population, residual

    population, residual = evaluate(multipliers)
    for _ in range(max_iters):
        if residual.abs().amax() < tolerance:
            break
        jacobian = (population.unsqueeze(1) * moment_matrix.unsqueeze(0)) @ moment_matrix.transpose(0, 1)
        jacobian = jacobian + 1.0e-8 * identity
        delta = torch.linalg.solve(jacobian, residual.unsqueeze(-1)).squeeze(-1)
        multipliers = multipliers - delta
        population, residual = evaluate(multipliers)
    return population


class NeurDEConstrainedGeq(nn.Module):
    def __init__(
        self,
        alpha_layer,
        phi_layer,
        activation,
        cv,
        logit_clip=15.0,
        newton_iters=20,
        newton_tolerance=1.0e-6,
    ):
        super().__init__()
        self.alpha = DenseNet(alpha_layer, activation)
        self.phi = DenseNet(phi_layer, activation)
        self.cv = cv
        self.logit_clip = float(logit_clip)
        self.newton_iters = int(newton_iters)
        self.newton_tolerance = float(newton_tolerance)

        for layer in reversed(self.alpha.layers):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                break

    def forward(self, u0, grid, geq_base):
        flat_macro = _flatten_macro_state(u0)
        alpha_coeffs = self.alpha(flat_macro)
        basis_coeffs = self.phi(grid)
        logits = torch.einsum("bi,ni->bn", alpha_coeffs, basis_coeffs).clamp(
            min=-self.logit_clip,
            max=self.logit_clip,
        )
        base_flat = _flatten_population(geq_base).to(device=logits.device, dtype=logits.dtype)
        moment_matrix = _build_moment_matrix(grid).to(device=logits.device, dtype=logits.dtype)
        correction = _project_to_nonconserved(logits, moment_matrix)
        log_base = torch.log(base_flat.clamp_min(1.0e-12)) + correction
        target_moments = _geq_targets(flat_macro, self.cv)
        return _solve_constrained_geq(
            log_base,
            grid,
            target_moments,
            self.newton_iters,
            self.newton_tolerance,
        )

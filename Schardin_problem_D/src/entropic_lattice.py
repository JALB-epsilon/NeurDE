from dataclasses import dataclass

import torch

from .multinv import _multinv_torch


@dataclass
class MultispeedLattice:
    cx_base: torch.Tensor
    cy_base: torch.Tensor
    cx: torch.Tensor
    cy: torch.Tensor
    speed_sq: torch.Tensor
    weights: torch.Tensor
    opp: torch.Tensor

    @property
    def q(self):
        return int(self.cx.numel())


def build_d2q49_lattice(device, dtype, shift_x=0.0, shift_y=0.0):
    one_d_speeds = torch.arange(-3, 4, device=device, dtype=dtype)
    one_d_weights = torch.tensor(
        [
            8.12130e-04,
            2.69382e-02,
            2.33915e-01,
            4.76670e-01,
            2.33915e-01,
            2.69382e-02,
            8.12130e-04,
        ],
        device=device,
        dtype=dtype,
    )
    grid_y, grid_x = torch.meshgrid(one_d_speeds, one_d_speeds, indexing="ij")
    weights_y, weights_x = torch.meshgrid(one_d_weights, one_d_weights, indexing="ij")

    cx_base = grid_x.reshape(-1)
    cy_base = grid_y.reshape(-1)
    weights = (weights_x * weights_y).reshape(-1)
    cx = cx_base + torch.as_tensor(float(shift_x), device=device, dtype=dtype)
    cy = cy_base + torch.as_tensor(float(shift_y), device=device, dtype=dtype)
    speed_sq = cx * cx + cy * cy

    opp = torch.empty_like(cx_base, dtype=torch.long)
    speed_pairs = {(int(x.item()), int(y.item())): idx for idx, (x, y) in enumerate(zip(cx_base, cy_base))}
    for idx, (x, y) in enumerate(zip(cx_base, cy_base)):
        opp[idx] = speed_pairs[(-int(x.item()), -int(y.item()))]

    return MultispeedLattice(
        cx_base=cx_base,
        cy_base=cy_base,
        cx=cx,
        cy=cy,
        speed_sq=speed_sq,
        weights=weights,
        opp=opp,
    )


def _reshape_lambdas(lambdas, shape, device, dtype):
    if lambdas is None:
        return None
    if lambdas.dim() == 1:
        return lambdas.to(device=device, dtype=dtype).reshape(1, 4)
    return lambdas.to(device=device, dtype=dtype).reshape(4, -1).transpose(0, 1)


def compute_entropic_equilibrium(
    rho,
    ux,
    uy,
    T,
    lattice,
    lambdas=None,
    newton_iters=8,
    tolerance=1e-8,
):
    shape = rho.shape
    device = rho.device
    dtype = rho.dtype

    rho_flat = rho.reshape(-1).clamp_min(1e-8)
    ux_flat = ux.reshape(-1)
    uy_flat = uy.reshape(-1)
    T_flat = T.reshape(-1).clamp_min(1e-6)
    n_cells = rho_flat.numel()

    if n_cells == 0:
        empty_f = torch.zeros((lattice.q, 0), device=device, dtype=dtype)
        empty_l = torch.zeros((4, 0), device=device, dtype=dtype)
        return empty_f, empty_l

    basis = torch.stack(
        (
            torch.ones_like(lattice.cx),
            lattice.cx,
            lattice.cy,
            lattice.speed_sq,
        ),
        dim=0,
    ).to(device=device, dtype=dtype)
    weights = lattice.weights.to(device=device, dtype=dtype).unsqueeze(1)

    target = torch.stack(
        (
            rho_flat,
            rho_flat * ux_flat,
            rho_flat * uy_flat,
            rho_flat * (ux_flat * ux_flat + uy_flat * uy_flat + 2.0 * T_flat),
        ),
        dim=1,
    )

    lambda_flat = _reshape_lambdas(lambdas, shape, device, dtype)
    if lambda_flat is None or lambda_flat.shape[0] != n_cells:
        lambda_flat = torch.zeros((n_cells, 4), device=device, dtype=dtype)
        lambda_flat[:, 0] = torch.log(rho_flat)
        lambda_flat[:, 3] = -0.25 / T_flat.clamp_min(1e-2)

    basis_t = basis.transpose(0, 1)
    exponent_limit = 80.0
    delta_limit = 2.0
    eye = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)

    for _ in range(int(newton_iters)):
        exponent = torch.matmul(lambda_flat, basis).transpose(0, 1).clamp(min=-exponent_limit, max=exponent_limit)
        feq = weights * torch.exp(exponent)
        moments = torch.matmul(basis, feq).transpose(0, 1)
        residual = moments - target
        rel = residual.abs() / target.abs().clamp_min(1e-6)
        if torch.max(rel) < tolerance:
            break

        weighted_basis = feq.transpose(0, 1).unsqueeze(-1) * basis_t.unsqueeze(0)
        jacobian = torch.matmul(weighted_basis.transpose(1, 2), basis_t.unsqueeze(0))
        jacobian = jacobian + 1e-10 * eye
        delta = torch.bmm(_multinv_torch(jacobian), residual.unsqueeze(-1)).squeeze(-1)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=delta_limit, neginf=-delta_limit).clamp(
            min=-delta_limit, max=delta_limit
        )
        lambda_flat = lambda_flat - delta

    exponent = torch.matmul(lambda_flat, basis).transpose(0, 1).clamp(min=-exponent_limit, max=exponent_limit)
    feq = weights * torch.exp(exponent)
    feq = feq.clamp_min(1e-12)
    return feq.reshape(lattice.q, *shape), lambda_flat.transpose(0, 1).reshape(4, *shape)


def compute_geq_from_feq(feq, T, Cv, dimensions=2):
    internal_factor = max(2.0 * float(Cv) - float(dimensions), 0.0)
    return internal_factor * T.unsqueeze(0) * feq


def entropy(populations, weights):
    weights_view = weights.reshape(-1, *([1] * (populations.dim() - 1)))
    safe = populations.clamp_min(1e-20)
    return torch.sum(safe * (torch.log(safe) - torch.log(weights_view)), dim=0)


def estimate_entropic_alpha(populations, equilibrium, weights, beta, max_halves=8):
    beta_view = beta.unsqueeze(0)
    delta = beta_view * (equilibrium - populations)
    alpha = torch.full(beta.shape, 2.0, device=populations.device, dtype=populations.dtype)

    negative = delta < 0
    if torch.any(negative):
        alpha_cap = torch.where(negative, -0.999 * populations / delta.clamp_max(-1e-20), torch.full_like(delta, 1e9))
        alpha = torch.minimum(alpha, torch.amin(alpha_cap, dim=0))

    alpha = alpha.clamp(min=1e-3, max=2.0)
    h0 = entropy(populations, weights)
    for _ in range(max_halves):
        candidate = populations + alpha.unsqueeze(0) * delta
        h1 = entropy(candidate, weights)
        bad = (~torch.isfinite(h1)) | (h1 > h0)
        if not torch.any(bad):
            break
        alpha = torch.where(bad, 0.5 * alpha, alpha)
    return alpha.clamp(min=1e-3, max=2.0)

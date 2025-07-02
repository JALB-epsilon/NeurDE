import torch
import numpy as np
from .multinv import _multinv_torch

def levermore_Geq_torch(
    ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, device=None, use_sparse=False
):

    if use_sparse:
        print("Warning: Sparse inversion is not supported in PyTorch. Using dense batched inversion instead.")
    is_numpy = isinstance(ux, np.ndarray)

    # Determine target device
    if device is None:
        device = ux.device if not is_numpy else 'cpu'

    # --- Optimized Tensor Conversion ("If Needed") ---
    tensors = [torch.as_tensor(v, dtype=torch.float32, device=device)
               for v in (ex, ey, ux, uy, T, rho, khi, zetax, zetay)]
    ex, ey, ux, uy, T, rho, khi, zetax, zetay = tensors
    Cv = float(Cv)
    Qn = int(Qn)

    # Numerical stability
    T.masked_fill_(torch.abs(T) < 1e-6, 0)
    rho.masked_fill_(torch.abs(rho) < 1e-6, 0)

    Y, X = ux.shape
    ex = ex.squeeze()
    ey = ey.squeeze()

    uu = ux**2 + uy**2
    E = T * Cv + 0.5 * uu
    H = E + T

    w = torch.zeros((Qn, Y, X), device=device, dtype=torch.float32)
    w[:4, :, :] = (1 - T) * T * 0.5
    w[4:8, :, :] = T**2 * 0.25
    w[8, :, :] = (1 - T)**2

    max_iterations = 10
    tol = 1e-6
    # Batch-aware convergence mask: True = not yet converged
    mask = torch.ones((Y, X), dtype=torch.bool, device=device)
    for _ in range(max_iterations):
        if not mask.any():  # Early exit if all points have converged
            break

        # Only update non-converged points
        khi.masked_fill_((torch.abs(khi) < tol) & mask, 0)
        zetax.masked_fill_((torch.abs(zetax) < tol) & mask, 0)
        zetay.masked_fill_((torch.abs(zetay) < tol) & mask, 0)

        exponent = khi[None, :, :] + zetax[None, :, :] * ex[:, None, None] + zetay[None, :, :] * ey[:, None, None]
        f = w * torch.exp(exponent)

        # Precompute constant terms
        f_sum = f.sum(dim=0)  # Sum over velocities
        f_ex = torch.einsum("q,qyx->yx", ex, f)  # Weighted sum with ex
        f_ey = torch.einsum("q,qyx->yx", ey, f)  # Weighted sum with ey

        # Construct residual vector F using precomputed terms
        F = torch.zeros((3, Y, X), dtype=torch.float32, device=device)
        F[0, :, :] = f_sum - 2 * E  # Equation for density
        F[1, :, :] = f_ex - 2 * ux * H  # Equation for x-momentum
        F[2, :, :] = f_ey - 2 * uy * H  # Equation for y-momentum

        J = torch.zeros((3, 3, Y, X), dtype=torch.float32, device=device)
        J[0, 0, :, :] = f_sum  # dF_0/d(khi)
        J[0, 1, :, :] = f_ex  # dF_0/d(zetax)
        J[0, 2, :, :] = f_ey  # dF_0/d(zetay)
        J[1, 0, :, :], J[2, 0, :, :] = J[0, 1, :, :], J[0, 2, :, :]
        J[1, 1, :, :] = torch.einsum("q,qyx->yx", ex**2, f)
        J[1, 2, :, :] = torch.einsum("q,qyx->yx", ex * ey, f)
        J[2, 1, :, :] = J[1, 2, :, :]
        J[2, 2, :, :] = torch.einsum("q,qyx->yx", ey**2, f)

        IJ = _multinv_torch(J, device=device)

        khi1, zetax1, zetay1 = khi.clone(), zetax.clone(), zetay.clone()
        delta = torch.einsum('ijyx,jyx->iyx', IJ, F)

        # Perform in-place updates only on the non-converged points.
        # This can be more memory-efficient than creating new tensors with torch.where.
        khi[mask] -= delta[0][mask]
        zetax[mask] -= delta[1][mask]
        zetay[mask] -= delta[2][mask]
        # Compute convergence for this step
        dkhi = torch.abs(khi - khi1)
        dzetax = torch.abs(zetax - zetax1)
        dzetay = torch.abs(zetay - zetay1)
        # Update mask: a point remains unconverged if it was previously unconverged
        # AND its change in this step was greater than the tolerance.
        mask &= (dkhi > tol) | (dzetax > tol) | (dzetay > tol)

    Feq = w * rho[None, :, :] * torch.exp(khi[None, :, :] + zetax[None, :, :] * ex[:, None, None] + zetay[None, :, :] * ey[:, None, None])

    if is_numpy:
        return Feq.cpu().numpy(), khi.cpu().numpy(), zetax.cpu().numpy(), zetay.cpu().numpy()
    else:
        return Feq, khi, zetax, zetay




def levermore_Geq_BCs_torch(
    ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, row, col, device=None
):
    """
    PyTorch version of levermore_Geq_BCs with in-place update of full arrays at boundary points.
    """
    is_numpy = isinstance(ux, np.ndarray)
    if device is None:
        device = ux.device if not is_numpy else 'cpu'
    # Convert all to torch tensors
    tensors = [torch.as_tensor(v, dtype=torch.float32, device=device) for v in (ex, ey, ux, uy, T, rho, khi, zetax, zetay)]
    ex, ey, ux, uy, T, rho, khi, zetax, zetay = tensors
    Cv = float(Cv)
    Qn = int(Qn)
    eps = 1e-12

    # Ensure all fields are 2D (Y, X)
    if ux.ndim == 1:
        N = ux.numel()
        Y = int(torch.sqrt(torch.tensor(N)))
        X = N // Y
        ux = ux.view(Y, X)
        uy = uy.view(Y, X)
        T = T.view(Y, X)
        rho = rho.view(Y, X)
        khi = khi.view(Y, X)
        zetax = zetax.view(Y, X)
        zetay = zetay.view(Y, X)

    row = torch.as_tensor(row, dtype=torch.long, device=device)
    col = torch.as_tensor(col, dtype=torch.long, device=device)
    R = row.shape[0]
    ex = ex.squeeze()
    ey = ey.squeeze()

    # Precompute common terms
    uu = ux**2 + uy**2
    E = T * Cv + 0.5 * uu
    H = E + T

    # Precompute weights only at boundary points
    w = torch.zeros((Qn, R), device=device, dtype=torch.float32)
    one_minus_T = 1 - T[row, col]
    w[:4, :] = one_minus_T * T[row, col] * 0.5
    w[4:8, :] = T[row, col] ** 2 * 0.25
    w[8, :] = one_minus_T ** 2

    # Gather boundary values
    ux_b = ux[row, col]
    uy_b = uy[row, col]
    T_b = T[row, col]
    rho_b = rho[row, col]
    khi_b = khi[row, col]
    zetax_b = zetax[row, col]
    zetay_b = zetay[row, col]

    max_iterations = 20
    tol = 1e-6
    mask = torch.ones(R, dtype=torch.bool, device=device)
    for _ in range(max_iterations):
        # Numerical stability
        khi_b = torch.where(torch.abs(khi_b) < tol, torch.zeros_like(khi_b), khi_b)
        zetax_b = torch.where(torch.abs(zetax_b) < tol, torch.zeros_like(zetax_b), zetax_b)
        zetay_b = torch.where(torch.abs(zetay_b) < tol, torch.zeros_like(zetay_b), zetay_b)

        exponent = khi_b[None, :] + zetax_b[None, :] * ex[:, None] + zetay_b[None, :] * ey[:, None]
        f = w * torch.exp(exponent)

        f_sum = f.sum(dim=0)
        ex_dot_f = torch.matmul(ex, f)
        ey_dot_f = torch.matmul(ey, f)

        F = torch.zeros((3, R), device=device, dtype=torch.float32)
        F[0, :] = f_sum - 2 * E[row, col]
        F[1, :] = ex_dot_f - 2 * ux_b * H[row, col]
        F[2, :] = ey_dot_f - 2 * uy_b * H[row, col]

        J = torch.zeros((3, 3, R), device=device, dtype=torch.float32)
        J[0, 0, :] = f_sum
        J[0, 1, :] = ex_dot_f
        J[0, 2, :] = ey_dot_f
        J[1, 0, :] = ex_dot_f
        J[1, 1, :] = torch.matmul(ex ** 2, f)
        J[1, 2, :] = torch.matmul(ex * ey, f)
        J[2, 0, :] = ey_dot_f
        J[2, 1, :] = J[1, 2, :]
        J[2, 2, :] = torch.matmul(ey ** 2, f)

        IJ = _multinv_torch(J, device=device)
        khi1 = khi_b.clone()
        zetax1 = zetax_b.clone()
        zetay1 = zetay_b.clone()

        delta = torch.einsum('ijr,jr->ir', IJ, F)
        khi_b = khi_b - delta[0]
        zetax_b = zetax_b - delta[1]
        zetay_b = zetay_b - delta[2]

        # Convergence check
        dkhi = torch.abs((khi_b - khi1) / (khi1 + eps))
        dzetax = torch.abs((zetax_b - zetax1) / (zetax1 + eps))
        dzetay = torch.abs((zetay_b - zetay1) / (zetay1 + eps))
        mx = torch.max(torch.stack([dkhi, dzetax, dzetay]))
        if mx < 1e-6:
            break

    # Update the full arrays at the boundary points
    khi[row, col] = khi_b
    zetax[row, col] = zetax_b
    zetay[row, col] = zetay_b

    exponent = khi_b[None, :] + zetax_b[None, :] * ex[:, None] + zetay_b[None, :] * ey[:, None]
    Feq = w * rho_b[None, :] * torch.exp(exponent)

    if is_numpy:
        Feq = Feq.cpu().numpy()
        khi = khi.cpu().numpy()
        zetax = zetax.cpu().numpy()
        zetay = zetay.cpu().numpy()
    return Feq, khi, zetax, zetay


def levermore_Geq_Obs_torch(
    ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, Obs, device=None
):
    """
    Calculates Levermore equilibrium for observed (obstacle) points, in-place update of full arrays at Obs.
    """
    is_numpy = isinstance(ux, np.ndarray)
    if device is None:
        device = ux.device if not is_numpy else 'cpu'
    # Convert all to torch tensors
    tensors = [torch.as_tensor(v, dtype=torch.float32, device=device)
               for v in (ex, ey, ux, uy, T, rho, khi, zetax, zetay)]
    ex, ey, ux, uy, T, rho, khi, zetax, zetay = tensors
    Obs = torch.as_tensor(Obs, dtype=torch.bool, device=device)
    Cv = float(Cv)
    Qn = int(Qn)
    eps = 1e-12

    # Numerical stability
    ux = torch.where(torch.abs(ux) < 1e-5, torch.zeros_like(ux), ux)
    uy = torch.where(torch.abs(uy) < 1e-5, torch.zeros_like(uy), uy)
    T = torch.where(torch.abs(T) < 1e-5, torch.zeros_like(T), T)
    rho = torch.where(torch.abs(rho) < 1e-5, torch.zeros_like(rho), rho)

    ex = ex.squeeze()
    ey = ey.squeeze()

    # Gather obstacle points
    idx = Obs.nonzero(as_tuple=True)
    L = idx[0].numel()
    ux_b = ux[Obs]
    uy_b = uy[Obs]
    T_b = T[Obs]
    rho_b = rho[Obs]
    khi_b = khi[Obs]
    zetax_b = zetax[Obs]
    zetay_b = zetay[Obs]

    uu = ux_b ** 2 + uy_b ** 2
    E = T_b * Cv + uu / 2
    H = E + T_b

    w = torch.zeros((Qn, L), device=device, dtype=ux.dtype)
    one_minus_T = 1 - T_b
    w[:4, :] = one_minus_T * T_b * 0.5
    w[4:8, :] = T_b ** 2 * 0.25
    w[8, :] = one_minus_T ** 2

    max_iterations = 20
    tol = 1e-6
    for _ in range(max_iterations):
        # Numerical stability
        khi_b = torch.where(torch.abs(khi_b) < tol, torch.zeros_like(khi_b), khi_b)
        zetax_b = torch.where(torch.abs(zetax_b) < tol, torch.zeros_like(zetax_b), zetax_b)
        zetay_b = torch.where(torch.abs(zetay_b) < tol, torch.zeros_like(zetay_b), zetay_b)

        exponent = khi_b[None, :] + zetax_b[None, :] * ex[:, None] + zetay_b[None, :] * ey[:, None]
        f = w * torch.exp(exponent)

        f_sum = f.sum(dim=0)
        ex_dot_f = torch.matmul(ex, f)
        ey_dot_f = torch.matmul(ey, f)

        F = torch.zeros((3, L), device=device, dtype=ux.dtype)
        F[0, :] = f_sum - 2 * E
        F[1, :] = ex_dot_f - 2 * ux_b * H
        F[2, :] = ey_dot_f - 2 * uy_b * H

        J = torch.zeros((3, 3, L), device=device, dtype=ux.dtype)
        J[0, 0, :] = f_sum
        J[0, 1, :] = ex_dot_f
        J[0, 2, :] = ey_dot_f
        J[1, 0, :] = ex_dot_f
        J[1, 1, :] = torch.matmul(ex ** 2, f)
        J[1, 2, :] = torch.matmul(ex * ey, f)
        J[2, 0, :] = ey_dot_f
        J[2, 1, :] = J[1, 2, :]
        J[2, 2, :] = torch.matmul(ey ** 2, f)

        IJ = _multinv_torch(J, device=device)
        khi1 = khi_b.clone()
        zetax1 = zetax_b.clone()
        zetay1 = zetay_b.clone()

        delta = torch.einsum('ijr,jr->ir', IJ, F)
        khi_b = khi_b - delta[0]
        zetax_b = zetax_b - delta[1]
        zetay_b = zetay_b - delta[2]

        # Convergence check
        dkhi = torch.abs((khi_b - khi1) / (khi1 + eps))
        dzetax = torch.abs((zetax_b - zetax1) / (zetax1 + eps))
        dzetay = torch.abs((zetay_b - zetay1) / (zetay1 + eps))
        mx = torch.max(torch.stack([dkhi, dzetax, dzetay]))
        if mx < 1e-6:
            break

    # Update the full arrays at the obstacle points
    khi[Obs] = khi_b
    zetax[Obs] = zetax_b
    zetay[Obs] = zetay_b

    exponent = khi_b[None, :] + zetax_b[None, :] * ex[:, None] + zetay_b[None, :] * ey[:, None]
    Feq = w * rho_b[None, :] * torch.exp(exponent)

    if is_numpy:
        Feq = Feq.cpu().numpy()
        khi = khi.cpu().numpy()
        zetax = zetax.cpu().numpy()
        zetay = zetay.cpu().numpy()
    return Feq, khi, zetax, zetay
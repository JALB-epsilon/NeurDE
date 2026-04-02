import numpy as np
import torch

from .multinv import _multinv_torch, multinv


def _levermore_weights_flat(T_flat, Qn):
    w = torch.zeros((Qn, T_flat.numel()), device=T_flat.device, dtype=T_flat.dtype)
    if Qn == 0:
        return w

    one_minus_T = 1.0 - T_flat
    if Qn >= 9:
        w[:4] = (one_minus_T * T_flat * 0.5).unsqueeze(0).expand(4, -1)
        w[4:8] = (T_flat * T_flat * 0.25).unsqueeze(0).expand(4, -1)
        w[8] = one_minus_T * one_minus_T
        if Qn > 9:
            w[9:] = 0.1
    else:
        count = min(4, Qn)
        w[:count] = (one_minus_T * T_flat * 0.5).unsqueeze(0).expand(count, -1)
    return w


def _reshape_equilibrium(Feq_flat, leading_shape, spatial_shape, Qn):
    reshaped = Feq_flat.transpose(0, 1).reshape(*leading_shape, *spatial_shape, Qn)
    dims = list(range(reshaped.dim()))
    dims.insert(len(leading_shape), dims.pop(-1))
    return reshaped.permute(*dims)


def _solve_levermore_flat(ex, ey, ux_flat, uy_flat, T_flat, rho_flat, Cv_tensor, Qn, khi_flat, zetax_flat, zetay_flat):
    if ux_flat.numel() == 0:
        empty = torch.zeros((Qn, 0), device=ux_flat.device, dtype=ux_flat.dtype)
        return empty, khi_flat, zetax_flat, zetay_flat

    T_flat = T_flat.clamp(min=1e-6)
    rho_flat = rho_flat.clamp(min=1e-6)

    uu = ux_flat * ux_flat + uy_flat * uy_flat
    E = T_flat * Cv_tensor + 0.5 * uu
    H = E + T_flat

    w = _levermore_weights_flat(T_flat, Qn)
    ex_column = ex.unsqueeze(1)
    ey_column = ey.unsqueeze(1)
    ex_sq = ex * ex
    ey_sq = ey * ey
    ex_ey = ex * ey

    exponent_limit = 50.0
    delta_limit = 1.0

    for _ in range(20):
        khi_flat = khi_flat.clamp(min=-1e6, max=1e6)
        zetax_flat = zetax_flat.clamp(min=-1e6, max=1e6)
        zetay_flat = zetay_flat.clamp(min=-1e6, max=1e6)

        exponent = khi_flat.unsqueeze(0) + zetax_flat.unsqueeze(0) * ex_column + zetay_flat.unsqueeze(0) * ey_column
        exponent = exponent.clamp(min=-exponent_limit, max=exponent_limit)
        f = w * torch.exp(exponent)

        f_sum = f.sum(dim=0)
        f_ex = torch.sum(ex_column * f, dim=0)
        f_ey = torch.sum(ey_column * f, dim=0)

        F = torch.stack(
            [
                f_sum - 2.0 * E,
                f_ex - 2.0 * ux_flat * H,
                f_ey - 2.0 * uy_flat * H,
            ],
            dim=-1,
        )

        J = torch.zeros((ux_flat.numel(), 3, 3), device=ux_flat.device, dtype=ux_flat.dtype)
        J[:, 0, 0] = f_sum
        J[:, 0, 1] = f_ex
        J[:, 0, 2] = f_ey
        J[:, 1, 0] = f_ex
        J[:, 1, 1] = torch.sum(ex_sq.unsqueeze(1) * f, dim=0)
        J[:, 1, 2] = torch.sum(ex_ey.unsqueeze(1) * f, dim=0)
        J[:, 2, 0] = f_ey
        J[:, 2, 1] = J[:, 1, 2]
        J[:, 2, 2] = torch.sum(ey_sq.unsqueeze(1) * f, dim=0)

        delta = torch.bmm(_multinv_torch(J), F.unsqueeze(-1)).squeeze(-1)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=delta_limit, neginf=-delta_limit)
        delta = delta.clamp(min=-delta_limit, max=delta_limit)

        khi_old = khi_flat.clone()
        zetax_old = zetax_flat.clone()
        zetay_old = zetay_flat.clone()

        khi_flat = torch.nan_to_num(khi_flat - delta[:, 0], nan=0.0, posinf=1e6, neginf=-1e6)
        zetax_flat = torch.nan_to_num(zetax_flat - delta[:, 1], nan=0.0, posinf=1e6, neginf=-1e6)
        zetay_flat = torch.nan_to_num(zetay_flat - delta[:, 2], nan=0.0, posinf=1e6, neginf=-1e6)

        max_delta = torch.max(
            torch.stack(
                [
                    torch.abs(khi_flat - khi_old).max(),
                    torch.abs(zetax_flat - zetax_old).max(),
                    torch.abs(zetay_flat - zetay_old).max(),
                ]
            )
        )
        if max_delta < 1e-6:
            break

    exponent = khi_flat.unsqueeze(0) + zetax_flat.unsqueeze(0) * ex_column + zetay_flat.unsqueeze(0) * ey_column
    exponent = exponent.clamp(min=-exponent_limit, max=exponent_limit)
    Feq = w * rho_flat.unsqueeze(0) * torch.exp(exponent)
    return Feq, khi_flat, zetax_flat, zetay_flat

def levermore_Geq(ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay):
    """Calculates the Levermore equilibrium distribution function (optimized)."""
    ux[np.abs(ux) < 1e-6] = 0
    uy[np.abs(uy) < 1e-6] = 0
    T[np.abs(T) < 1e-6] = 0
    rho[np.abs(rho) < 1e-6] = 0

    Y, X = ux.shape
    Qn = int(Qn)
    ex = ex.squeeze()
    ey = ey.squeeze()

    uu = ux**2 + uy**2
    E = T * Cv + 0.5 * uu
    H = E + T

    w = np.zeros((Qn, Y, X))  
    one_minus_T = 1 - T
    w[:4, :, :] = one_minus_T * T * 0.5
    w[4:8, :, :] = T**2 * 0.25
    w[8, :, :] = one_minus_T**2

    f = np.zeros((Qn, Y, X))
    F = np.zeros((3, Y, X))
    J = np.zeros((3, 3, Y, X))

    for _ in range(20):
        khi[np.abs(khi) < 1e-6] = 0
        zetax[np.abs(zetax) < 1e-6] = 0
        zetay[np.abs(zetay) < 1e-6] = 0

        f = w * np.exp(khi[None, :, :] + zetax[None, :, :] * ex[:, None, None] + zetay[None, :, :] * ey[:, None, None])

        F[0, :, :] = f.sum(axis=0) - 2 * E
        F[1, :, :] = (ex[:, None, None] * f).sum(axis=0) - 2 * ux * H
        F[2, :, :] = (ey[:, None, None] * f).sum(axis=0) - 2 * uy * H


        J[0, 0, :, :] = f.sum(axis=0)
        J[0, 1, :, :] = np.einsum("q,qyx->yx", ex, f)  
        J[0, 2, :, :] = np.einsum("q,qyx->yx", ey, f)
        J[1, 0, :, :] = J[0, 1, :, :]
        J[1, 1, :, :] = np.einsum("q,qyx->yx", ex**2, f)
        J[1, 2, :, :] = np.einsum("q,qyx->yx", ex * ey, f)
        J[2, 0, :, :] = J[0, 2, :, :]
        J[2, 1, :, :] = J[1, 2, :, :]
        J[2, 2, :, :] = np.einsum("q,qyx->yx", ey**2, f)
        
        IJ = multinv(J) 
        IJ = IJ.reshape((3,3,Y,X), order="F")

        khi1 = khi.copy()
        zetax1 = zetax.copy()
        zetay1 = zetay.copy()

        khi -= (IJ[0, 0] * F[0] + IJ[0, 1] * F[1] + IJ[0, 2] * F[2])
        zetax -= (IJ[1, 0] * F[0] + IJ[1, 1] * F[1] + IJ[1, 2] * F[2])
        zetay -= (IJ[2, 0] * F[0] + IJ[2, 1] * F[1] + IJ[2, 2] * F[2])

        dkhi = np.abs(khi - khi1)
        dzetax = np.abs(zetax - zetax1)
        dzetay = np.abs(zetay - zetay1)

        mx = np.max(np.array([np.max(dkhi), np.max(dzetax), np.max(dzetay)]))  
        if mx < 1e-6:
            break

    Feq = w * rho[None, :, :] * np.exp(khi[None, :, :] + zetax[None, :, :] * ex[:, None, None] + zetay[None, :, :] * ey[:, None, None])

    return Feq, khi, zetax, zetay


def levermore_Geq_BCs(ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, row, col):
    """Calculates Levermore equilibrium for boundary conditions (optimized)."""

    # Ensure numerical stability
    ux[np.abs(ux) < 1e-6] = 0
    uy[np.abs(uy) < 1e-6] = 0
    T[np.abs(T) < 1e-6] = 0
    rho[np.abs(rho) < 1e-6] = 0
    eps = 1e-12

    Y, X = ux.shape
    Qn = int(Qn)
    ex = ex.squeeze()
    ey = ey.squeeze()

    # Precompute common terms
    uu = ux**2 + uy**2
    E = T * Cv + 0.5 * uu
    H = E + T

    # Precompute weights
    w = np.zeros((Qn, Y, X))
    one_minus_T = (1 - T[row, col]) 
    w[:4, row, col] = one_minus_T * T[row, col] *0.5
    w[4:8, row, col] = T[row, col]**2 * 0.25
    w[8, row, col] = one_minus_T**2

    R = len(row)
    F = np.zeros((3, R))
    J = np.zeros((3, 3, R))
    # Newton-Raphson iteration
    for _ in range(20):
        khi[np.abs(khi) < 1e-6] = 0
        zetax[np.abs(zetax) < 1e-6] = 0
        zetay[np.abs(zetay) < 1e-6] = 0

        # Calculate exponential term
        exponent = khi[None, row, col] + zetax[None, row, col] * ex[:, None] + zetay[None, row, col] * ey[:, None]
        f = w[:, row, col] * np.exp(exponent)

        # Calculate F and J using vectorized operations
       
        F[0] = np.sum(f, axis=0) - 2 * E[row, col]
        #precompute ex and ey dot f 
        ex_dot_f = np.dot(ex, f)
        ey_dot_f = np.dot(ey, f)

        F[1] = ex_dot_f - 2 * ux[row, col] * H[row, col]
        F[2] = ey_dot_f - 2 * uy[row, col] * H[row, col]

        J[0, 0] = np.sum(f, axis=0)
        J[0, 1] = ex_dot_f
        J[0, 2] = ey_dot_f
        J[1, 0] = ex_dot_f
        J[1, 1] = np.dot(ex**2, f)
        J[1, 2] = np.dot(ex * ey, f)
        J[2, 0] = ey_dot_f
        J[2, 1] = J[1, 2]
        J[2, 2] = np.dot(ey**2, f)


        # Calculate inverse Jacobian and update parameters
        IJ = np.linalg.inv(J.transpose(2, 0, 1))  # Transpose to get correct shape for batched inverse
        d_params = -np.matmul(IJ, F).transpose(0, 2, 1)[:, 0, :]


        khi_old = khi[row, col].copy()
        zetax_old = zetax[row, col].copy()
        zetay_old = zetay[row, col].copy()

        khi[row, col] += d_params[:, 0]
        zetax[row, col] += d_params[:, 1]
        zetay[row, col] += d_params[:, 2]

        # Calculate convergence criteria
        dkhi = np.abs((khi[row, col] - khi_old) / (khi_old + eps))
        dzetax = np.abs((zetax[row, col] - zetax_old) / (zetax_old + eps))
        dzetay = np.abs((zetay[row, col] - zetay_old) / (zetay_old + eps))

        max_diff = max(np.max(dkhi), np.max(dzetax), np.max(dzetay))

        if max_diff < 1e-6:
            break

    # Calculate final equilibrium distribution
    exponent = khi[None, row, col] + zetax[None, row, col] * ex[:, None] + zetay[None, row, col] * ey[:, None]
    Feq = w[:, row, col] * rho[row, col] * np.exp(exponent)

    return Feq, khi, zetax, zetay


def levermore_Geq_Obs(ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, Obs):
    """Calculates Levermore equilibrium for observed points (optimized)."""
    ux[np.abs(ux) < 1e-5] = 0
    uy[np.abs(uy) < 1e-5] = 0
    T[np.abs(T) < 1e-5] = 0
    rho[np.abs(rho) < 1e-5] = 0

    ex = ex.squeeze()
    ey = ey.squeeze()
    eps = 1e-12

    uu = ux[Obs]**2 + uy[Obs]**2
    E = T[Obs] * Cv + uu / 2
    H = E + T[Obs]
    L = np.arange(len(uu))

    w = np.zeros((Qn, len(L)))
    one_minus_T = 1 - T[Obs]
    w[:4, L] = one_minus_T * T[Obs] * 0.5
    w[4:8, L] = T[Obs]**2 * 0.25
    w[8, L] = one_minus_T**2

    dkhi = np.zeros_like(khi, order='F')
    dzetax = np.zeros_like(zetax, order='F')
    dzetay = np.zeros_like(zetay, order='F')
    F = np.zeros((3, len(L)))
    J = np.zeros((3, 3, len(L)))
    for _ in range(20):
        khi[np.abs(khi) < 1e-5] = 0
        zetax[np.abs(zetax) < 1e-5] = 0
        zetay[np.abs(zetay) < 1e-5] = 0

        f = w * np.exp(khi[Obs] + zetax[None, Obs] * ex[:, None] + zetay[None, Obs] * ey[:, None])

        #precompute ex and ey dot f 
        ex_dot_f = np.dot(ex, f)
        ey_dot_f = np.dot(ey, f)

        F[0, L] = np.sum(f, axis=0) - 2 * E
        F[1, L] = ex_dot_f - 2 * ux[Obs] * H
        F[2, L] = ey_dot_f - 2 * uy[Obs] * H
        

        
        J[0, 0, L] = np.sum(f, axis=0)
        J[0, 1, L] = ex_dot_f
        J[0, 2, L] = ey_dot_f
        J[1, 0, L] = ex_dot_f
        J[1, 1, L] = np.dot(ex**2, f)
        J[1, 2, L] = np.dot(ex * ey, f)
        J[2, 0, L] = ey_dot_f
        J[2, 1, L] = J[1, 2, L]
        J[2, 2, L] = np.dot(ey**2, f)


        IJ = np.linalg.inv(J.transpose(2, 0, 1))

        d_params = -np.matmul(IJ, F).transpose(0, 2, 1)[:, 0, :]

        khi_old = khi[Obs].copy()
        zetax_old = zetax[Obs].copy()
        zetay_old = zetay[Obs].copy()

        khi[Obs] += d_params[:, 0] 
        zetax[Obs] += d_params[:, 1]
        zetay[Obs] += d_params[:, 2] 

        dkhi[Obs] = np.abs((khi[Obs] - khi_old) / (khi_old + eps))
        dzetax[Obs] = np.abs((zetax[Obs] - zetax_old) / (zetax_old + eps))
        dzetay[Obs] = np.abs((zetay[Obs] - zetay_old) / (zetay_old + eps))

        max_diff = np.max(np.stack((dkhi[Obs], dzetax[Obs], dzetay[Obs])), axis=0).max()

        if max_diff < 1e-6:
            break

    Feq = w * rho[None, Obs] * np.exp(khi[None, Obs] + zetax[None, Obs] * ex[:, None] + zetay[None, Obs] * ey[:, None])

    return Feq, khi, zetax, zetay




''''
def levermore_Geq_BCs(ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, row, col):
    """Calculates Levermore equilibrium for boundary conditions (optimized)."""

    # Ensure numerical stability
    ux[np.abs(ux) < 1e-5] = 0
    uy[np.abs(uy) < 1e-5] = 0
    T[np.abs(T) < 1e-5] = 0
    rho[np.abs(rho) < 1e-5] = 0
    eps = 1e-12

    Y, X = ux.shape
    Qn = int(Qn)
    ex = ex.squeeze()
    ey = ey.squeeze()

    # Precompute common terms
    uu = ux**2 + uy**2
    E = T * Cv + 0.5 * uu
    H = E + T

    # Precompute weights
    w = np.zeros((Qn, Y, X))
    w[:4, row, col] = (1 - T[row, col]) * T[row, col] / 2
    w[4:8, row, col] = T[row, col]**2 / 4
    w[8, row, col] = (1 - T[row, col])**2

    R = len(row)

    # Newton-Raphson iteration
    
    dkhi = np.zeros_like(khi, order='F')
    dzetax = np.zeros_like(zetax, order='F')
    dzetay = np.zeros_like(zetay, order='F')

    for _ in range(20):
        khi[np.abs(khi) < 1e-5] = 0
        zetax[np.abs(zetax) < 1e-5] = 0
        zetay[np.abs(zetay) < 1e-5] = 0

        # Calculate exponential term
        exponent = khi[None, row, col] + zetax[None, row, col] * ex[:, None] + zetay[None, row, col] * ey[:, None]
        f = w[:, row, col] * np.exp(exponent)

        # Calculate F and J using vectorized operations
        F = np.zeros((3, R))
        F[0] = np.sum(f, axis=0) - 2 * E[row, col]
        #precompute ex and ey dot f 
        ex_dot_f = np.dot(ex, f)
        ey_dot_f = np.dot(ey, f)

        F[1] = ex_dot_f - 2 * ux[row, col] * H[row, col]
        F[2] = ey_dot_f - 2 * uy[row, col] * H[row, col]

        J = np.zeros((3, 3, R))
        J[0, 0] = np.sum(f, axis=0)
        J[0, 1] = ex_dot_f
        J[0, 2] = ey_dot_f
        J[1, 0] = ex_dot_f
        J[1, 1] = np.dot(ex**2, f)
        J[1, 2] = np.dot(ex * ey, f)
        J[2, 0] = ey_dot_f
        J[2, 1] = np.dot(ey * ex, f)
        J[2, 2] = np.dot(ey**2, f)


        IJ = multinv(J)
        khi1 = khi.copy()
        zetax1 = zetax.copy() 
        zetay1 = zetay.copy()

        # newton Method to find root
        khi[row,col] = khi[row,col] -(IJ[0, 0] * F[0] + IJ[0, 1] * F[1] + IJ[0, 2] * F[2])       
        zetax[row,col] =zetax[row,col]- (IJ[1, 0] * F[0] + IJ[1, 1] * F[1] + IJ[1, 2] * F[2])
        zetay[row,col] =zetay[row,col]- (IJ[2, 0] * F[0] + IJ[2, 1] * F[1] + IJ[2, 2] * F[2]) 
        
        dkhi[row,col] = np.abs((khi[row,col] - khi1[row,col]) / (khi1[row,col]+eps))
        dzetax[row,col] = np.abs((zetax[row,col] - zetax1[row,col]) / (zetax1[row,col]+eps))
        dzetay[row,col] = np.abs((zetay[row,col] - zetay1[row,col]) / (zetay1[row,col]+eps))

        mkhi = np.max(np.abs(dkhi[row,col]))
        mzetax = np.max(np.abs(dzetax[row,col]))
        mzetay = np.max(np.abs(dzetay[row,col]))

        mx = max([mkhi, mzetax, mzetay])

        if mx < 1e-6:
            break

    # Calculate final equilibrium distribution
    exponent = khi[None, row, col] + zetax[None, row, col] * ex[:, None] + zetay[None, row, col] * ey[:, None]
    Feq = w[:, row, col] * rho[row, col] * np.exp(exponent)

    return Feq, khi, zetax, zetay


def levermore_Geq_Obs(ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, Obs):
    """Calculates Levermore equilibrium for observed points (optimized)."""
    ux[np.abs(ux) < 1e-5] = 0
    uy[np.abs(uy) < 1e-5] = 0
    T[np.abs(T) < 1e-5] = 0
    rho[np.abs(rho) < 1e-5] = 0

    ex = ex.squeeze()
    ey = ey.squeeze()
    eps = 1e-12

    uu = ux[Obs]**2 + uy[Obs]**2
    E = T[Obs] * Cv + uu / 2
    H = E + T[Obs]
    L = np.arange(len(uu))

    w = np.zeros((Qn, len(L)))
    one_minus_T = 1 - T[Obs]
    w[:4, L] = one_minus_T * T[Obs] * 0.5
    w[4:8, L] = T[Obs]**2 * 0.25
    w[8, L] = one_minus_T**2

    dkhi = np.zeros_like(khi, order='F')
    dzetax = np.zeros_like(zetax, order='F')
    dzetay = np.zeros_like(zetay, order='F')

    for _ in range(20):
        khi[np.abs(khi) < 1e-6] = 0
        zetax[np.abs(zetax) < 1e-6] = 0
        zetay[np.abs(zetay) < 1e-6] = 0

        f = w * np.exp(khi[Obs] + zetax[None, Obs] * ex[:, None] + zetay[None, Obs] * ey[:, None])

        #precompute ex and ey dot f 
        ex_dot_f = np.dot(ex, f)
        ey_dot_f = np.dot(ey, f)

        F = np.zeros((3, len(L)))
        F[0, L] = np.sum(f, axis=0) - 2 * E
        F[1, L] = ex_dot_f - 2 * ux[Obs] * H
        F[2, L] = ey_dot_f - 2 * uy[Obs] * H
        

        J = np.zeros((3, 3, len(L)))
        J[0, 0, L] = np.sum(f, axis=0)
        J[0, 1, L] = ex_dot_f
        J[0, 2, L] = ey_dot_f
        J[1, 0, L] = ex_dot_f
        J[1, 1, L] = np.dot(ex**2, f)
        J[1, 2, L] = np.dot(ex * ey, f)
        J[2, 0, L] = ey_dot_f
        J[2, 1, L] = np.dot(ey * ex, f)
        J[2, 2, L] = np.dot(ey**2, f)


        IJ = multinv(J)

        khi1 = khi[Obs].copy()
        zetax1 = zetax[Obs].copy()
        zetay1 = zetay[Obs].copy()

        sz = F[0, L].shape[0]

        # Pre-compute F slices (scalar values)
        F_0L = F[0, L]
        F_1L = F[1, L]
        F_2L = F[2, L]

        # Directly update khi, zetax, zetay using views (no new arrays created)
        khi[Obs] -= (IJ[0, 0, L].reshape(sz, order='F') * F_0L +
                    IJ[0, 1, L].reshape(sz, order='F') * F_1L +
                    IJ[0, 2, L].reshape(sz, order='F') * F_2L)

        zetax[Obs] -= (IJ[1, 0, L].reshape(sz, order='F') * F_0L +
                    IJ[1, 1, L].reshape(sz, order='F') * F_1L +
                    IJ[1, 2, L].reshape(sz, order='F') * F_2L)

        zetay[Obs] -= (IJ[2, 0, L].reshape(sz, order='F') * F_0L +
                    IJ[2, 1, L].reshape(sz, order='F') * F_1L +
                    IJ[2, 2, L].reshape(sz, order='F') * F_2L)

        # Calculate differences directly into dkhi, dzetax, dzetay
        np.abs((khi[Obs] - khi1) / (khi1 + eps), out=dkhi[Obs])
        np.abs((zetax[Obs] - zetax1) / (zetax1 + eps), out=dzetax[Obs])
        np.abs((zetay[Obs] - zetay1) / (zetay1 + eps), out=dzetay[Obs])

        # Calculate minimums directly
        mkhi = np.min(dkhi[Obs])
        mzetax = np.min(dzetax[Obs])
        mzetay = np.min(dzetay[Obs])



        mx = min(mkhi, mzetax, mzetay)

        if mx < 1e-6:
            break

    Feq = w * rho[None, Obs] * np.exp(khi[None, Obs] + zetax[None, Obs] * ex[:, None] + zetay[None, Obs] * ey[:, None])

    return Feq, khi, zetax, zetay
'''


def levermore_Geq_torch(
    ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, device=None, use_sparse=False
):
    """Torch version of the Levermore equilibrium Newton solve."""
    del use_sparse

    device = ux.device if device is None else device
    dtype = ux.dtype
    Cv_tensor = torch.as_tensor(Cv, device=device, dtype=dtype)

    ex = ex.flatten()[:Qn].to(device=device, dtype=dtype)
    ey = ey.flatten()[:Qn].to(device=device, dtype=dtype)
    khi = khi.to(device=device, dtype=dtype)
    zetax = zetax.to(device=device, dtype=dtype)
    zetay = zetay.to(device=device, dtype=dtype)

    leading_shape = ux.shape[:-2]
    spatial_shape = ux.shape[-2:]

    Feq_flat, khi_flat, zetax_flat, zetay_flat = _solve_levermore_flat(
        ex,
        ey,
        ux.reshape(-1),
        uy.reshape(-1),
        T.reshape(-1),
        rho.reshape(-1),
        Cv_tensor,
        Qn,
        khi.reshape(-1),
        zetax.reshape(-1),
        zetay.reshape(-1),
    )

    Feq = _reshape_equilibrium(Feq_flat, leading_shape, spatial_shape, Qn)
    return Feq, khi_flat.reshape_as(khi), zetax_flat.reshape_as(zetax), zetay_flat.reshape_as(zetay)


def levermore_Geq_BCs_torch(
    ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, row, col, device=None
):
    """Torch version of the boundary Levermore equilibrium Newton solve."""
    device = ux.device if device is None else device
    dtype = ux.dtype
    Cv_tensor = torch.as_tensor(Cv, device=device, dtype=dtype)

    ex = ex.flatten()[:Qn].to(device=device, dtype=dtype)
    ey = ey.flatten()[:Qn].to(device=device, dtype=dtype)
    row = torch.as_tensor(row, device=device, dtype=torch.long)
    if row.ndim == 0:
        row = row.unsqueeze(0)
    col = torch.as_tensor(col, device=device, dtype=torch.long)
    if col.ndim == 0:
        col = col.expand_as(row)

    khi = khi.to(device=device, dtype=dtype)
    zetax = zetax.to(device=device, dtype=dtype)
    zetay = zetay.to(device=device, dtype=dtype)

    selected_shape = ux[..., row, col].shape
    Feq_flat, khi_flat, zetax_flat, zetay_flat = _solve_levermore_flat(
        ex,
        ey,
        ux[..., row, col].reshape(-1),
        uy[..., row, col].reshape(-1),
        T[..., row, col].reshape(-1),
        rho[..., row, col].reshape(-1),
        Cv_tensor,
        Qn,
        khi[..., row, col].reshape(-1),
        zetax[..., row, col].reshape(-1),
        zetay[..., row, col].reshape(-1),
    )

    khi[..., row, col] = khi_flat.reshape(selected_shape)
    zetax[..., row, col] = zetax_flat.reshape(selected_shape)
    zetay[..., row, col] = zetay_flat.reshape(selected_shape)

    Feq = _reshape_equilibrium(Feq_flat, selected_shape[:-1], (selected_shape[-1],), Qn)
    return Feq, khi, zetax, zetay


def levermore_Geq_Obs_torch(
    ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, Obs, device=None
):
    """Torch version of the obstacle Levermore equilibrium Newton solve."""
    device = ux.device if device is None else device
    dtype = ux.dtype
    Cv_tensor = torch.as_tensor(Cv, device=device, dtype=dtype)

    ex = ex.flatten()[:Qn].to(device=device, dtype=dtype)
    ey = ey.flatten()[:Qn].to(device=device, dtype=dtype)
    khi = khi.to(device=device, dtype=dtype)
    zetax = zetax.to(device=device, dtype=dtype)
    zetay = zetay.to(device=device, dtype=dtype)

    Obs = torch.as_tensor(Obs, device=device, dtype=torch.bool)
    selected_shape = ux[..., Obs].shape

    Feq_flat, khi_flat, zetax_flat, zetay_flat = _solve_levermore_flat(
        ex,
        ey,
        ux[..., Obs].reshape(-1),
        uy[..., Obs].reshape(-1),
        T[..., Obs].reshape(-1),
        rho[..., Obs].reshape(-1),
        Cv_tensor,
        Qn,
        khi[..., Obs].reshape(-1),
        zetax[..., Obs].reshape(-1),
        zetay[..., Obs].reshape(-1),
    )

    khi[..., Obs] = khi_flat.reshape(selected_shape)
    zetax[..., Obs] = zetax_flat.reshape(selected_shape)
    zetay[..., Obs] = zetay_flat.reshape(selected_shape)

    Feq = _reshape_equilibrium(Feq_flat, selected_shape[:-1], (selected_shape[-1],), Qn)
    return Feq, khi, zetax, zetay

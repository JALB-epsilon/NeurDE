import numpy as np
import torch

from .multinv import _multinv_torch, multinv

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

    T = T.clamp(min=1e-6)
    rho = rho.clamp(min=1e-6)

    Y, X = ux.shape
    uu = ux * ux + uy * uy
    E = T * Cv_tensor + 0.5 * uu
    H = E + T

    w = torch.zeros((Qn, Y, X), device=device, dtype=dtype)
    if Qn >= 9:
        one_minus_T = 1.0 - T
        w[:4] = one_minus_T * T * 0.5
        w[4:8] = T * T * 0.25
        w[8] = one_minus_T * one_minus_T
        if Qn > 9:
            w[9:] = 0.1
    else:
        w[: min(4, Qn)] = (1.0 - T) * T * 0.5

    ex_expanded = ex[:, None, None]
    ey_expanded = ey[:, None, None]
    ex_sq = ex * ex
    ey_sq = ey * ey
    ex_ey = ex * ey

    F = torch.zeros((3, Y, X), device=device, dtype=dtype)
    J = torch.zeros((3, 3, Y, X), device=device, dtype=dtype)

    for _ in range(20):
        khi = khi.clamp(min=-1e6, max=1e6)
        zetax = zetax.clamp(min=-1e6, max=1e6)
        zetay = zetay.clamp(min=-1e6, max=1e6)

        exponent = khi[None, :, :] + zetax[None, :, :] * ex_expanded + zetay[None, :, :] * ey_expanded
        f = w * torch.exp(exponent)

        f_sum = f.sum(dim=0)
        f_ex = torch.einsum("q,qyx->yx", ex, f)
        f_ey = torch.einsum("q,qyx->yx", ey, f)

        F[0] = f_sum - 2.0 * E
        F[1] = f_ex - 2.0 * ux * H
        F[2] = f_ey - 2.0 * uy * H

        J[0, 0] = f_sum
        J[0, 1] = f_ex
        J[0, 2] = f_ey
        J[1, 0] = f_ex
        J[1, 1] = torch.einsum("q,qyx->yx", ex_sq, f)
        J[1, 2] = torch.einsum("q,qyx->yx", ex_ey, f)
        J[2, 0] = f_ey
        J[2, 1] = J[1, 2]
        J[2, 2] = torch.einsum("q,qyx->yx", ey_sq, f)

        IJ = _multinv_torch(J.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)

        khi_old = khi.clone()
        zetax_old = zetax.clone()
        zetay_old = zetay.clone()

        khi = khi - (IJ[0, 0] * F[0] + IJ[0, 1] * F[1] + IJ[0, 2] * F[2])
        zetax = zetax - (IJ[1, 0] * F[0] + IJ[1, 1] * F[1] + IJ[1, 2] * F[2])
        zetay = zetay - (IJ[2, 0] * F[0] + IJ[2, 1] * F[1] + IJ[2, 2] * F[2])

        max_delta = torch.max(
            torch.stack(
                [
                    torch.abs(khi - khi_old).max(),
                    torch.abs(zetax - zetax_old).max(),
                    torch.abs(zetay - zetay_old).max(),
                ]
            )
        )
        if max_delta < 1e-6:
            break

    Feq = w * rho[None, :, :] * torch.exp(
        khi[None, :, :] + zetax[None, :, :] * ex_expanded + zetay[None, :, :] * ey_expanded
    )
    return Feq, khi, zetax, zetay


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

    uu = ux * ux + uy * uy
    E = T * Cv_tensor + 0.5 * uu
    H = E + T

    ux_b = ux[row, col]
    uy_b = uy[row, col]
    T_b = T[row, col].clamp(min=1e-6)
    rho_b = rho[row, col].clamp(min=1e-6)
    E_b = E[row, col]
    H_b = H[row, col]
    khi_b = khi[row, col].clone()
    zetax_b = zetax[row, col].clone()
    zetay_b = zetay[row, col].clone()

    R = row.shape[0]
    w = torch.zeros((Qn, R), device=device, dtype=dtype)
    one_minus_T = 1.0 - T_b
    w[:4] = (one_minus_T * T_b * 0.5).unsqueeze(0).expand(4, -1)
    w[4:8] = (T_b * T_b * 0.25).unsqueeze(0).expand(4, -1)
    w[8] = one_minus_T * one_minus_T

    ex_sq = ex * ex
    ey_sq = ey * ey
    ex_ey = ex * ey

    for _ in range(20):
        khi_b = khi_b.clamp(min=-1e6, max=1e6)
        zetax_b = zetax_b.clamp(min=-1e6, max=1e6)
        zetay_b = zetay_b.clamp(min=-1e6, max=1e6)

        exponent = khi_b.unsqueeze(0) + zetax_b.unsqueeze(0) * ex.unsqueeze(1) + zetay_b.unsqueeze(0) * ey.unsqueeze(1)
        f = w * torch.exp(exponent)

        f_sum = f.sum(dim=0)
        f_ex = torch.sum(ex.unsqueeze(1) * f, dim=0)
        f_ey = torch.sum(ey.unsqueeze(1) * f, dim=0)

        F = torch.stack(
            [
                f_sum - 2.0 * E_b,
                f_ex - 2.0 * ux_b * H_b,
                f_ey - 2.0 * uy_b * H_b,
            ]
        )

        J = torch.zeros((R, 3, 3), device=device, dtype=dtype)
        J[:, 0, 0] = f_sum
        J[:, 0, 1] = f_ex
        J[:, 0, 2] = f_ey
        J[:, 1, 0] = f_ex
        J[:, 1, 1] = torch.sum(ex_sq.unsqueeze(1) * f, dim=0)
        J[:, 1, 2] = torch.sum(ex_ey.unsqueeze(1) * f, dim=0)
        J[:, 2, 0] = f_ey
        J[:, 2, 1] = J[:, 1, 2]
        J[:, 2, 2] = torch.sum(ey_sq.unsqueeze(1) * f, dim=0)

        J_inv = _multinv_torch(J)

        khi_old = khi_b.clone()
        zetax_old = zetax_b.clone()
        zetay_old = zetay_b.clone()

        delta = torch.bmm(J_inv, F.transpose(0, 1).unsqueeze(-1)).squeeze(-1)
        khi_b = khi_b - delta[:, 0]
        zetax_b = zetax_b - delta[:, 1]
        zetay_b = zetay_b - delta[:, 2]

        max_delta = torch.max(
            torch.stack(
                [
                    torch.abs(khi_b - khi_old).max(),
                    torch.abs(zetax_b - zetax_old).max(),
                    torch.abs(zetay_b - zetay_old).max(),
                ]
            )
        )
        if max_delta < 1e-6:
            break

    khi[row, col] = khi_b
    zetax[row, col] = zetax_b
    zetay[row, col] = zetay_b

    Feq = w * rho_b.unsqueeze(0) * torch.exp(
        khi_b.unsqueeze(0) + zetax_b.unsqueeze(0) * ex.unsqueeze(1) + zetay_b.unsqueeze(0) * ey.unsqueeze(1)
    )
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
    ux_b = ux[Obs]
    uy_b = uy[Obs]
    T_b = T[Obs].clamp(min=1e-6)
    rho_b = rho[Obs].clamp(min=1e-6)
    khi_b = khi[Obs].clone()
    zetax_b = zetax[Obs].clone()
    zetay_b = zetay[Obs].clone()

    L = ux_b.shape[0]
    if L == 0:
        return torch.zeros((Qn, 0), device=device, dtype=dtype), khi, zetax, zetay

    uu = ux_b * ux_b + uy_b * uy_b
    E = T_b * Cv_tensor + 0.5 * uu
    H = E + T_b

    w = torch.zeros((Qn, L), device=device, dtype=dtype)
    one_minus_T = 1.0 - T_b
    w[:4] = (one_minus_T * T_b * 0.5).unsqueeze(0).expand(4, -1)
    w[4:8] = (T_b * T_b * 0.25).unsqueeze(0).expand(4, -1)
    w[8] = one_minus_T * one_minus_T

    ex_sq = ex * ex
    ey_sq = ey * ey
    ex_ey = ex * ey

    for _ in range(20):
        khi_b = khi_b.clamp(min=-1e6, max=1e6)
        zetax_b = zetax_b.clamp(min=-1e6, max=1e6)
        zetay_b = zetay_b.clamp(min=-1e6, max=1e6)

        exponent = khi_b.unsqueeze(0) + zetax_b.unsqueeze(0) * ex.unsqueeze(1) + zetay_b.unsqueeze(0) * ey.unsqueeze(1)
        f = w * torch.exp(exponent)

        f_sum = f.sum(dim=0)
        f_ex = torch.sum(ex.unsqueeze(1) * f, dim=0)
        f_ey = torch.sum(ey.unsqueeze(1) * f, dim=0)

        F = torch.stack(
            [
                f_sum - 2.0 * E,
                f_ex - 2.0 * ux_b * H,
                f_ey - 2.0 * uy_b * H,
            ]
        )

        J = torch.zeros((L, 3, 3), device=device, dtype=dtype)
        J[:, 0, 0] = f_sum
        J[:, 0, 1] = f_ex
        J[:, 0, 2] = f_ey
        J[:, 1, 0] = f_ex
        J[:, 1, 1] = torch.sum(ex_sq.unsqueeze(1) * f, dim=0)
        J[:, 1, 2] = torch.sum(ex_ey.unsqueeze(1) * f, dim=0)
        J[:, 2, 0] = f_ey
        J[:, 2, 1] = J[:, 1, 2]
        J[:, 2, 2] = torch.sum(ey_sq.unsqueeze(1) * f, dim=0)

        J_inv = _multinv_torch(J)

        khi_old = khi_b.clone()
        zetax_old = zetax_b.clone()
        zetay_old = zetay_b.clone()

        delta = torch.bmm(J_inv, F.transpose(0, 1).unsqueeze(-1)).squeeze(-1)
        khi_b = khi_b - delta[:, 0]
        zetax_b = zetax_b - delta[:, 1]
        zetay_b = zetay_b - delta[:, 2]

        max_delta = torch.max(
            torch.stack(
                [
                    torch.abs(khi_b - khi_old).max(),
                    torch.abs(zetax_b - zetax_old).max(),
                    torch.abs(zetay_b - zetay_old).max(),
                ]
            )
        )
        if max_delta < 1e-6:
            break

    khi[Obs] = khi_b
    zetax[Obs] = zetax_b
    zetay[Obs] = zetay_b

    Feq = w * rho_b.unsqueeze(0) * torch.exp(
        khi_b.unsqueeze(0) + zetax_b.unsqueeze(0) * ex.unsqueeze(1) + zetay_b.unsqueeze(0) * ey.unsqueeze(1)
    )
    return Feq, khi, zetax, zetay

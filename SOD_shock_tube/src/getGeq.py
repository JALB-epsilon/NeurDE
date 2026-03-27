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
    ONE9 = np.ones((1, Qn))
    ex = ex.squeeze()
    ey = ey.squeeze()

    uu = ux**2 + uy**2
    E = T * Cv + 0.5 * uu
    H = E + T

    w = np.zeros((Qn, Y, X))  
    w[:4, :, :] = (1 - T) * T * 0.5
    w[4:8, :, :] = T**2 * 0.25
    w[8, :, :] = (1 - T)**2

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


def levermore_Geq_torch(
    ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, device=None, use_sparse=False
):
    """Torch version of the Levermore Newton solve used by compiled workflows."""
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

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

    for _ in range(20):
        khi_flat = khi_flat.clamp(min=-1e6, max=1e6)
        zetax_flat = zetax_flat.clamp(min=-1e6, max=1e6)
        zetay_flat = zetay_flat.clamp(min=-1e6, max=1e6)

        exponent = khi_flat.unsqueeze(0) + zetax_flat.unsqueeze(0) * ex_column + zetay_flat.unsqueeze(0) * ey_column
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

        khi_old = khi_flat.clone()
        zetax_old = zetax_flat.clone()
        zetay_old = zetay_flat.clone()

        khi_flat = khi_flat - delta[:, 0]
        zetax_flat = zetax_flat - delta[:, 1]
        zetay_flat = zetay_flat - delta[:, 2]

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

    Feq = w * rho_flat.unsqueeze(0) * torch.exp(
        khi_flat.unsqueeze(0) + zetax_flat.unsqueeze(0) * ex_column + zetay_flat.unsqueeze(0) * ey_column
    )
    return Feq, khi_flat, zetax_flat, zetay_flat

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

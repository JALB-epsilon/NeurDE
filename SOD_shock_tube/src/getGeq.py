import numpy as np
import torch
from .multinv import _multinv_torch

'''def levermore_Geq(ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay):
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

    return Feq, khi, zetax, zetay'''


def levermore_Geq_torch(
   ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, device=None, use_sparse=False
):

    if use_sparse:
        print("Warning: Sparse inversion is not supported in PyTorch. Using dense batched inversion instead.")
    is_numpy = isinstance(ux, np.ndarray)

    # Determine target device
    if device is None:
        device = ux.device if not is_numpy else 'cpu'


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
    
    # Use a fixed number of iterations with masked updates
    converged = torch.zeros((Y, X), dtype=torch.bool, device=device)
    
    for iteration in range(max_iterations):
        # Only update non-converged points
        active_mask = ~converged
        
        khi_masked = torch.where(active_mask & (torch.abs(khi) < tol), 0, khi)
        zetax_masked = torch.where(active_mask & (torch.abs(zetax) < tol), 0, zetax)
        zetay_masked = torch.where(active_mask & (torch.abs(zetay) < tol), 0, zetay)

        exponent = khi_masked[None, :, :] + zetax_masked[None, :, :] * ex[:, None, None] + zetay_masked[None, :, :] * ey[:, None, None]
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

        khi_old, zetax_old, zetay_old = khi.clone(), zetax.clone(), zetay.clone()
        delta = torch.einsum('ijyx,jyx->iyx', IJ, F)
        
        # Perform updates only on active points
        khi = torch.where(active_mask, khi - delta[0], khi)
        zetax = torch.where(active_mask, zetax - delta[1], zetax)
        zetay = torch.where(active_mask, zetay - delta[2], zetay)

        # Update convergence status
        dkhi = torch.abs(khi - khi_old)
        dzetax = torch.abs(zetax - zetax_old)
        dzetay = torch.abs(zetay - zetay_old)
        
        # Points converge if all deltas are below tolerance
        newly_converged = (dkhi <= tol) & (dzetax <= tol) & (dzetay <= tol)
        converged = converged | newly_converged

    Feq = w * rho[None, :, :] * torch.exp(khi[None, :, :] + zetax[None, :, :] * ex[:, None, None] + zetay[None, :, :] * ey[:, None, None])

    if is_numpy:
        return Feq.cpu().numpy(), khi.cpu().numpy(), zetax.cpu().numpy(), zetay.cpu().numpy()
    else:
        return Feq, khi, zetax, zetay
import numpy as np
import torch
from .multinv import _multinv_torch


def levermore_Geq_torch(
    ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, device=None, use_sparse=False
):
    """Optimized version for maximum speed with torch.compile"""
    
    # Convert scalar to tensor once
    Cv_tensor = torch.tensor(Cv, device=device, dtype=ux.dtype)
    
    # More efficient numerical stability - clamp is compile-friendly
    T = T.clamp(min=1e-6)
    rho = rho.clamp(min=1e-6)

    Y, X = ux.shape
    
    # Ensure ex, ey are 1D - more efficient shape handling
    ex = ex.flatten()[:Qn]
    ey = ey.flatten()[:Qn]
    
    # Pre-compute constants that don't change
    uu = ux * ux + uy * uy  # Faster than ux**2 + uy**2
    E = T * Cv_tensor + 0.5 * uu
    H = E + T

    # More efficient weight initialization
    w = torch.zeros((Qn, Y, X), device=device, dtype=ux.dtype)
    if Qn >= 9:
        T_term = (1 - T) * T * 0.5
        T_sq_term = T * T * 0.25
        T_sq_minus_term = (1 - T) * (1 - T)
        
        w[:4] = T_term
        w[4:8] = T_sq_term
        w[8] = T_sq_minus_term
        if Qn > 9:
            w[9:] = 0.1
    else:
        w[:min(4, Qn)] = (1 - T) * T * 0.5

    # Pre-compute ex and ey expansions for better memory access
    ex_expanded = ex[:, None, None]  # Shape: (Qn, 1, 1)
    ey_expanded = ey[:, None, None]  # Shape: (Qn, 1, 1)
    
    # Pre-compute ex**2, ey**2, ex*ey for Jacobian - avoid recomputing
    ex_sq = ex * ex
    ey_sq = ey * ey
    ex_ey = ex * ey

    # Constants for convergence
    max_iterations = 20
    tol = 1e-6
    conv_tol = 1e-6
    
    # Pre-allocate tensors to avoid repeated allocation
    F = torch.zeros((3, Y, X), device=device, dtype=ux.dtype)
    J = torch.zeros((3, 3, Y, X), device=device, dtype=ux.dtype)
    
    for iteration in range(max_iterations):
        # More efficient small value handling - clamp is compile-friendly
        khi = khi.clamp(min=-1e6, max=1e6)  # Prevent overflow instead of zeroing
        zetax = zetax.clamp(min=-1e6, max=1e6)
        zetay = zetay.clamp(min=-1e6, max=1e6)

        # Compute exponent more efficiently
        exponent = khi[None, :, :] + zetax[None, :, :] * ex_expanded + zetay[None, :, :] * ey_expanded
        
        # Compute distribution function
        f = w * torch.exp(exponent)

        # Compute moments - use more efficient einsum patterns
        f_sum = f.sum(dim=0)
        f_ex = torch.einsum("q,qyx->yx", ex, f)
        f_ey = torch.einsum("q,qyx->yx", ey, f)

        # Construct residual vector F - reuse pre-allocated tensor
        F[0] = f_sum - 2 * E
        F[1] = f_ex - 2 * ux * H
        F[2] = f_ey - 2 * uy * H

        # Construct Jacobian matrix - reuse pre-allocated tensor
        J[0, 0] = f_sum
        J[0, 1] = f_ex
        J[0, 2] = f_ey
        J[1, 0] = f_ex
        J[1, 1] = torch.einsum("q,qyx->yx", ex_sq, f)
        J[1, 2] = torch.einsum("q,qyx->yx", ex_ey, f)
        J[2, 0] = f_ey
        J[2, 1] = J[1, 2]  # Reuse computation
        J[2, 2] = torch.einsum("q,qyx->yx", ey_sq, f)

        # More efficient matrix operations
        J_batch = J.permute(2, 3, 0, 1)  # (Y, X, 3, 3)
        
        # Optimized matrix inverse
        IJ_batch = _multinv_torch(J_batch)
        IJ = IJ_batch.permute(2, 3, 0, 1)  # Back to (3, 3, Y, X)

        # Store old values more efficiently
        khi_old = khi.clone()
        zetax_old = zetax.clone()
        zetay_old = zetay.clone()
        
        # Newton step - use fused operations
        delta_khi = IJ[0, 0] * F[0] + IJ[0, 1] * F[1] + IJ[0, 2] * F[2]
        delta_zetax = IJ[1, 0] * F[0] + IJ[1, 1] * F[1] + IJ[1, 2] * F[2]
        delta_zetay = IJ[2, 0] * F[0] + IJ[2, 1] * F[1] + IJ[2, 2] * F[2]
        
        khi -= delta_khi
        zetax -= delta_zetax
        zetay -= delta_zetay

        # More efficient convergence check
        max_delta = torch.max(torch.stack([
            torch.abs(khi - khi_old).max(),
            torch.abs(zetax - zetax_old).max(),
            torch.abs(zetay - zetay_old).max()
        ]))
        
        if max_delta < conv_tol:
            break

    # Final computation
    final_exponent = khi[None, :, :] + zetax[None, :, :] * ex_expanded + zetay[None, :, :] * ey_expanded
    Feq = w * rho[None, :, :] * torch.exp(final_exponent)

    return Feq, khi, zetax, zetay
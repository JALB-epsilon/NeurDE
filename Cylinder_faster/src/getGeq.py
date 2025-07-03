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

def levermore_Geq_BCs_torch(
    ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, row, col, device=None
):
    """
    Optimized PyTorch version of levermore_Geq_BCs for torch.compile.
    """
    # Convert scalars to tensors once
    Cv_tensor = torch.tensor(Cv, device=device, dtype=ex.dtype)
    
    # Ensure all fields are 2D (Y, X) - more efficient reshaping
    if ux.ndim == 1:
        N = ux.numel()
        Y = int(torch.sqrt(torch.tensor(N, dtype=torch.float32)))
        X = N // Y
        # Reshape all tensors in one go
        shape = (Y, X)
        ux = ux.view(shape)
        uy = uy.view(shape)
        T = T.view(shape)
        rho = rho.view(shape)
        khi = khi.view(shape)
        zetax = zetax.view(shape)
        zetay = zetay.view(shape)

    R = row.shape[0]
    ex = ex.squeeze()
    ey = ey.squeeze()

    # Precompute common terms
    uu = ux * ux + uy * uy  # More efficient than **2
    E = T * Cv_tensor + 0.5 * uu
    H = E + T

    # Gather boundary values once
    ux_b = ux[row, col]
    uy_b = uy[row, col]
    T_b = T[row, col]
    rho_b = rho[row, col]
    khi_b = khi[row, col].clone()  # Clone to avoid in-place issues
    zetax_b = zetax[row, col].clone()
    zetay_b = zetay[row, col].clone()
    E_b = E[row, col]
    H_b = H[row, col]

    # Precompute weights - vectorized and more efficient
    one_minus_T = 1.0 - T_b
    T_squared = T_b * T_b
    
    # Create weight tensor efficiently
    w = torch.zeros((Qn, R), device=device, dtype=T_b.dtype)
    w_factor = one_minus_T * T_b * 0.5
    w[:4] = w_factor.unsqueeze(0).expand(4, -1)
    w[4:8] = (T_squared * 0.25).unsqueeze(0).expand(4, -1)
    w[8] = one_minus_T * one_minus_T

    # Precompute ex and ey powers for efficiency
    ex_sq = ex * ex
    ey_sq = ey * ey
    ex_ey = ex * ey
    
    # Constants
    tol = 1e-6
    max_iterations = 20
    convergence_tol = 1e-6
    
    # Newton-Raphson iteration
    for iteration in range(max_iterations):
        # Numerical stability - avoid branching with torch.where
        khi_b = torch.where(torch.abs(khi_b) < tol, torch.zeros_like(khi_b), khi_b)
        zetax_b = torch.where(torch.abs(zetax_b) < tol, torch.zeros_like(zetax_b), zetax_b)
        zetay_b = torch.where(torch.abs(zetay_b) < tol, torch.zeros_like(zetay_b), zetay_b)

        # Compute exponent more efficiently
        exponent = khi_b.unsqueeze(0) + zetax_b.unsqueeze(0) * ex.unsqueeze(1) + zetay_b.unsqueeze(0) * ey.unsqueeze(1)
        f = w * torch.exp(exponent)

        # Compute sums efficiently
        f_sum = f.sum(dim=0)
        ex_dot_f = torch.sum(ex.unsqueeze(1) * f, dim=0)
        ey_dot_f = torch.sum(ey.unsqueeze(1) * f, dim=0)

        # Residual vector
        F = torch.stack([
            f_sum - 2.0 * E_b,
            ex_dot_f - 2.0 * ux_b * H_b,
            ey_dot_f - 2.0 * uy_b * H_b
        ])

        # Jacobian matrix - compute all elements efficiently
        ex_sq_dot_f = torch.sum(ex_sq.unsqueeze(1) * f, dim=0)
        ey_sq_dot_f = torch.sum(ey_sq.unsqueeze(1) * f, dim=0)
        ex_ey_dot_f = torch.sum(ex_ey.unsqueeze(1) * f, dim=0)

        # Build Jacobian more efficiently
        J = torch.zeros((R, 3, 3), device=device, dtype=f.dtype)
        J[:, 0, 0] = f_sum
        J[:, 0, 1] = ex_dot_f
        J[:, 0, 2] = ey_dot_f
        J[:, 1, 0] = ex_dot_f
        J[:, 1, 1] = ex_sq_dot_f
        J[:, 1, 2] = ex_ey_dot_f
        J[:, 2, 0] = ey_dot_f
        J[:, 2, 1] = ex_ey_dot_f
        J[:, 2, 2] = ey_sq_dot_f

        # Batch matrix inversion using optimized function
        J_inv = _multinv_torch(J)
        
        # Store old values for convergence check
        khi_old = khi_b.clone()
        zetax_old = zetax_b.clone()
        zetay_old = zetay_b.clone()

        # Newton step - vectorized matrix multiplication
        delta = torch.bmm(J_inv, F.T.unsqueeze(-1)).squeeze(-1)
        
        khi_b -= delta[:, 0]
        zetax_b -= delta[:, 1]
        zetay_b -= delta[:, 2]

        # Convergence check - more efficient
        changes = torch.stack([
            torch.abs(khi_b - khi_old).max(),
            torch.abs(zetax_b - zetax_old).max(),
            torch.abs(zetay_b - zetay_old).max()
        ])
        if changes.max() < convergence_tol:
            break

    # Update the full arrays at the boundary points
    khi[row, col] = khi_b
    zetax[row, col] = zetax_b
    zetay[row, col] = zetay_b

    # Final computation
    exponent_final = khi_b.unsqueeze(0) + zetax_b.unsqueeze(0) * ex.unsqueeze(1) + zetay_b.unsqueeze(0) * ey.unsqueeze(1)
    Feq = w * rho_b.unsqueeze(0) * torch.exp(exponent_final)

    return Feq, khi, zetax, zetay


def levermore_Geq_Obs_torch(
    ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, Obs, device=None
):
    """
    Optimized Levermore equilibrium for obstacle points, optimized for torch.compile.
    """
    # Convert scalar to tensor once
    Cv_tensor = torch.tensor(Cv, device=device, dtype=ux.dtype)
    
    # Numerical stability - apply once at the beginning
    tol_stability = 1e-5
    ux = torch.where(torch.abs(ux) < tol_stability, torch.zeros_like(ux), ux)
    uy = torch.where(torch.abs(uy) < tol_stability, torch.zeros_like(uy), uy)
    T = torch.where(torch.abs(T) < tol_stability, torch.zeros_like(T), T)
    rho = torch.where(torch.abs(rho) < tol_stability, torch.zeros_like(rho), rho)

    ex = ex.squeeze()
    ey = ey.squeeze()

    # Gather obstacle points - avoid .item() for torch.compile
    ux_b = ux[Obs]
    L = ux_b.shape[0]  # Get number of obstacle points without .item()
    uy_b = uy[Obs]
    T_b = T[Obs]
    rho_b = rho[Obs]
    khi_b = khi[Obs].clone()  # Clone to avoid in-place issues
    zetax_b = zetax[Obs].clone()
    zetay_b = zetay[Obs].clone()

    # Precompute common terms
    uu = ux_b * ux_b + uy_b * uy_b  # More efficient than **2
    E = T_b * Cv_tensor + uu * 0.5
    H = E + T_b

    # Precompute weights more efficiently
    one_minus_T = 1.0 - T_b
    T_squared = T_b * T_b
    
    w = torch.zeros((Qn, L), device=device, dtype=ux.dtype)
    w_factor = one_minus_T * T_b * 0.5
    w[:4] = w_factor.unsqueeze(0).expand(4, -1)
    w[4:8] = (T_squared * 0.25).unsqueeze(0).expand(4, -1)
    w[8] = one_minus_T * one_minus_T

    # Precompute ex and ey powers for efficiency
    ex_sq = ex * ex
    ey_sq = ey * ey
    ex_ey = ex * ey
    
    # Constants
    tol = 1e-6
    max_iterations = 20
    convergence_tol = 1e-6
    
    # Newton-Raphson iteration
    for iteration in range(max_iterations):
        # Numerical stability
        khi_b = torch.where(torch.abs(khi_b) < tol, torch.zeros_like(khi_b), khi_b)
        zetax_b = torch.where(torch.abs(zetax_b) < tol, torch.zeros_like(zetax_b), zetax_b)
        zetay_b = torch.where(torch.abs(zetay_b) < tol, torch.zeros_like(zetay_b), zetay_b)

        # Compute exponent more efficiently
        exponent = khi_b.unsqueeze(0) + zetax_b.unsqueeze(0) * ex.unsqueeze(1) + zetay_b.unsqueeze(0) * ey.unsqueeze(1)
        f = w * torch.exp(exponent)

        # Compute sums efficiently
        f_sum = f.sum(dim=0)
        ex_dot_f = torch.sum(ex.unsqueeze(1) * f, dim=0)
        ey_dot_f = torch.sum(ey.unsqueeze(1) * f, dim=0)

        # Residual vector
        F = torch.stack([
            f_sum - 2.0 * E,
            ex_dot_f - 2.0 * ux_b * H,
            ey_dot_f - 2.0 * uy_b * H
        ])

        # Jacobian matrix - compute all elements efficiently
        ex_sq_dot_f = torch.sum(ex_sq.unsqueeze(1) * f, dim=0)
        ey_sq_dot_f = torch.sum(ey_sq.unsqueeze(1) * f, dim=0)
        ex_ey_dot_f = torch.sum(ex_ey.unsqueeze(1) * f, dim=0)

        # Build Jacobian more efficiently
        J = torch.zeros((L, 3, 3), device=device, dtype=ux.dtype)
        J[:, 0, 0] = f_sum
        J[:, 0, 1] = ex_dot_f
        J[:, 0, 2] = ey_dot_f
        J[:, 1, 0] = ex_dot_f
        J[:, 1, 1] = ex_sq_dot_f
        J[:, 1, 2] = ex_ey_dot_f
        J[:, 2, 0] = ey_dot_f
        J[:, 2, 1] = ex_ey_dot_f
        J[:, 2, 2] = ey_sq_dot_f

        # Batch matrix inversion using optimized function
        J_inv = _multinv_torch(J)
        
        # Store old values for convergence check
        khi_old = khi_b.clone()
        zetax_old = zetax_b.clone()
        zetay_old = zetay_b.clone()

        # Newton step - vectorized matrix multiplication
        delta = torch.bmm(J_inv, F.T.unsqueeze(-1)).squeeze(-1)
        
        khi_b -= delta[:, 0]
        zetax_b -= delta[:, 1]
        zetay_b -= delta[:, 2]

        # Convergence check - more efficient
        changes = torch.stack([
            torch.abs(khi_b - khi_old).max(),
            torch.abs(zetax_b - zetax_old).max(),
            torch.abs(zetay_b - zetay_old).max()
        ])
        if changes.max() < convergence_tol:
            break

    # Update the full arrays at the obstacle points
    khi[Obs] = khi_b
    zetax[Obs] = zetax_b
    zetay[Obs] = zetay_b

    # Final computation
    exponent_final = khi_b.unsqueeze(0) + zetax_b.unsqueeze(0) * ex.unsqueeze(1) + zetay_b.unsqueeze(0) * ey.unsqueeze(1)
    Feq = w * rho_b.unsqueeze(0) * torch.exp(exponent_final)

    return Feq, khi, zetax, zetay
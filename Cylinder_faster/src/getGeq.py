import numpy as np
import torch
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

    # Convert all inputs to tensors (use default dtype)
    tensors = [torch.as_tensor(v, device=device)
               for v in (ex, ey, ux, uy, T, rho, khi, zetax, zetay)]
    ex, ey, ux, uy, T, rho, khi, zetax, zetay = tensors
    Cv = float(Cv)
    Qn = int(Qn)

    # Numerical stability - use clamp instead of masked_fill for safety
    # Set very small values to zero for numerical stability (as in original numpy code)
    T = torch.where(torch.abs(T) < 1e-6, torch.zeros_like(T), T)
    rho = torch.where(torch.abs(rho) < 1e-6, torch.zeros_like(rho), rho)

    Y, X = ux.shape
    
    # Ensure ex, ey have the right shape
    if ex.dim() > 1:
        ex = ex.squeeze()
    if ey.dim() > 1:
        ey = ey.squeeze()
    
    # Ensure ex, ey are 1D with length Qn
    if ex.numel() != Qn:
        raise ValueError(f"ex must have {Qn} elements, got {ex.numel()}")
    if ey.numel() != Qn:
        raise ValueError(f"ey must have {Qn} elements, got {ey.numel()}")

    # Compute macroscopic quantities
    uu = ux**2 + uy**2
    E = T * Cv + 0.5 * uu
    H = E + T

    # Initialize weights - this seems to be a specific model choice
    w = torch.zeros((Qn, Y, X), device=device)
    
    # Fill weights based on original logic (adjust if needed for your specific model)
    if Qn >= 9:
        w[:4, :, :] = (1 - T) * T * 0.5
        w[4:8, :, :] = T**2 * 0.25
        w[8, :, :] = (1 - T)**2
        # Handle remaining weights if Qn > 9
        if Qn > 9:
            w[9:, :, :] = 0.1  # Default small value
    else:
        # Fallback for smaller Qn
        w[:min(4, Qn), :, :] = (1 - T) * T * 0.5

    max_iterations = 20
    tol = 1e-6
    
    for iteration in range(max_iterations):
        # Apply small value thresholding (matching original)
        khi = torch.where(torch.abs(khi) < tol, torch.zeros_like(khi), khi)
        zetax = torch.where(torch.abs(zetax) < tol, torch.zeros_like(zetax), zetax)
        zetay = torch.where(torch.abs(zetay) < tol, torch.zeros_like(zetay), zetay)

        # Compute exponent: shape should be (Qn, Y, X)
        exponent = (khi[None, :, :] + 
                   zetax[None, :, :] * ex[:, None, None] + 
                   zetay[None, :, :] * ey[:, None, None])
        
        # Compute distribution function
        f = w * torch.exp(exponent)

        # Compute moments
        f_sum = f.sum(dim=0)  # Sum over velocities: (Y, X)
        f_ex = torch.einsum("q,qyx->yx", ex, f)  # Weighted sum with ex
        f_ey = torch.einsum("q,qyx->yx", ey, f)  # Weighted sum with ey

        # Construct residual vector F
        F = torch.zeros((3, Y, X), device=device)
        F[0, :, :] = f_sum - 2 * E  # Mass conservation
        F[1, :, :] = f_ex - 2 * ux * H  # X-momentum conservation  
        F[2, :, :] = f_ey - 2 * uy * H  # Y-momentum conservation

        # Construct Jacobian matrix
        J = torch.zeros((3, 3, Y, X), device=device)
        
        # First row: derivatives of mass equation
        J[0, 0, :, :] = f_sum  # dF_0/d(khi)
        J[0, 1, :, :] = f_ex   # dF_0/d(zetax)
        J[0, 2, :, :] = f_ey   # dF_0/d(zetay)
        
        # Second row: derivatives of x-momentum equation
        J[1, 0, :, :] = f_ex   # dF_1/d(khi)  
        J[1, 1, :, :] = torch.einsum("q,qyx->yx", ex**2, f)  # dF_1/d(zetax)
        J[1, 2, :, :] = torch.einsum("q,qyx->yx", ex * ey, f)  # dF_1/d(zetay)
        
        # Third row: derivatives of y-momentum equation
        J[2, 0, :, :] = f_ey   # dF_2/d(khi)
        J[2, 1, :, :] = J[1, 2, :, :]  # dF_2/d(zetax)
        J[2, 2, :, :] = torch.einsum("q,qyx->yx", ey**2, f)  # dF_2/d(zetay)

        # Transpose J to get proper shape (Y, X, 3, 3) for batch inversion
        J_transposed = J.permute(2, 3, 0, 1)  # (Y, X, 3, 3)
        
        # Compute inverse Jacobian
        IJ_transposed = _multinv_torch(J_transposed, device=device)  # (Y, X, 3, 3)
        
        # Transpose back to (3, 3, Y, X)
        IJ = IJ_transposed.permute(2, 3, 0, 1)

        # Store old values for convergence check
        khi_old, zetax_old, zetay_old = khi.clone(), zetax.clone(), zetay.clone()
        
        # Compute Newton step (matching original calculation)
        khi -= (IJ[0, 0] * F[0] + IJ[0, 1] * F[1] + IJ[0, 2] * F[2])
        zetax -= (IJ[1, 0] * F[0] + IJ[1, 1] * F[1] + IJ[1, 2] * F[2])
        zetay -= (IJ[2, 0] * F[0] + IJ[2, 1] * F[1] + IJ[2, 2] * F[2])

        # Check convergence (matching original)
        dkhi = torch.abs(khi - khi_old)
        dzetax = torch.abs(zetax - zetax_old)
        dzetay = torch.abs(zetay - zetay_old)
        
        # Check max change (matching original logic) - only break if allowed
        mx = torch.max(torch.stack([dkhi.max(), dzetax.max(), dzetay.max()]))
        if mx < 1e-6:
            break

    # Compute final equilibrium distribution
    final_exponent = (khi[None, :, :] + 
                     zetax[None, :, :] * ex[:, None, None] + 
                     zetay[None, :, :] * ey[:, None, None])
    
    Feq = w * rho[None, :, :] * torch.exp(final_exponent)

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
    tensors = [torch.as_tensor(v, device=device) for v in (ex, ey, ux, uy, T, rho, khi, zetax, zetay)]
    ex, ey, ux, uy, T, rho, khi, zetax, zetay = tensors
    Cv = float(Cv)
    Qn = int(Qn)

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
    w = torch.zeros((Qn, R), device=device)
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
    
    for iteration in range(max_iterations):
        # Numerical stability
        khi_b = torch.where(torch.abs(khi_b) < tol, torch.zeros_like(khi_b), khi_b)
        zetax_b = torch.where(torch.abs(zetax_b) < tol, torch.zeros_like(zetax_b), zetax_b)
        zetay_b = torch.where(torch.abs(zetay_b) < tol, torch.zeros_like(zetay_b), zetay_b)

        exponent = khi_b[None, :] + zetax_b[None, :] * ex[:, None] + zetay_b[None, :] * ey[:, None]
        f = w * torch.exp(exponent)

        f_sum = f.sum(dim=0)
        ex_dot_f = torch.matmul(ex, f)
        ey_dot_f = torch.matmul(ey, f)

        F = torch.zeros((3, R), device=device)
        F[0, :] = f_sum - 2 * E[row, col]
        F[1, :] = ex_dot_f - 2 * ux_b * H[row, col]
        F[2, :] = ey_dot_f - 2 * uy_b * H[row, col]

        J = torch.zeros((3, 3, R), device=device)
        J[0, 0, :] = f_sum
        J[0, 1, :] = ex_dot_f
        J[0, 2, :] = ey_dot_f
        J[1, 0, :] = ex_dot_f
        J[1, 1, :] = torch.matmul(ex ** 2, f)
        J[1, 2, :] = torch.matmul(ex * ey, f)
        J[2, 0, :] = ey_dot_f
        J[2, 1, :] = J[1, 2, :]
        J[2, 2, :] = torch.matmul(ey ** 2, f)

        # Transpose J to get proper shape for batch inversion
        J_transposed = J.permute(2, 0, 1)  # (R, 3, 3)
        IJ_transposed = _multinv_torch(J_transposed, device=device)
        IJ = IJ_transposed.permute(1, 2, 0)  # Back to (3, 3, R)
        
        # Store old values for convergence check
        khi_old, zetax_old, zetay_old = khi_b.clone(), zetax_b.clone(), zetay_b.clone()

        # Compute Newton step
        khi_b -= (IJ[0, 0] * F[0] + IJ[0, 1] * F[1] + IJ[0, 2] * F[2])
        zetax_b -= (IJ[1, 0] * F[0] + IJ[1, 1] * F[1] + IJ[1, 2] * F[2])
        zetay_b -= (IJ[2, 0] * F[0] + IJ[2, 1] * F[1] + IJ[2, 2] * F[2])

        # Convergence check
        dkhi = torch.abs(khi_b - khi_old).max()
        dzetax = torch.abs(zetax_b - zetax_old).max()
        dzetay = torch.abs(zetay_b - zetay_old).max()
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
    tensors = [torch.as_tensor(v, device=device)
               for v in (ex, ey, ux, uy, T, rho, khi, zetax, zetay)]
    ex, ey, ux, uy, T, rho, khi, zetax, zetay = tensors
    Obs = torch.as_tensor(Obs, dtype=torch.bool, device=device)
    Cv = float(Cv)
    Qn = int(Qn)

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
    
    for iteration in range(max_iterations):
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

        # Transpose J to get proper shape for batch inversion
        J_transposed = J.permute(2, 0, 1)  # (L, 3, 3)
        IJ_transposed = _multinv_torch(J_transposed, device=device)
        IJ = IJ_transposed.permute(1, 2, 0)  # Back to (3, 3, L)
        
        # Store old values for convergence check
        khi_old, zetax_old, zetay_old = khi_b.clone(), zetax_b.clone(), zetay_b.clone()

        # Compute Newton step
        khi_b -= (IJ[0, 0] * F[0] + IJ[0, 1] * F[1] + IJ[0, 2] * F[2])
        zetax_b -= (IJ[1, 0] * F[0] + IJ[1, 1] * F[1] + IJ[1, 2] * F[2])
        zetay_b -= (IJ[2, 0] * F[0] + IJ[2, 1] * F[1] + IJ[2, 2] * F[2])

        # Convergence check
        dkhi = torch.abs(khi_b - khi_old).max()
        dzetax = torch.abs(zetax_b - zetax_old).max()
        dzetay = torch.abs(zetay_b - zetay_old).max()
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
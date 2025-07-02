import numpy as np
import torch
from .multinv import _multinv_torch



def levermore_Geq_torch(
    ex, ey, ux, uy, T, rho, Cv, Qn, khi, zetax, zetay, device=None, use_sparse=False, allow_early_termination=True
):
    if use_sparse:
        print("Warning: Sparse inversion is not supported in PyTorch. Using dense batched inversion instead.")
    
    is_numpy = isinstance(ux, np.ndarray)

    # Determine target device
    if device is None:
        device = ux.device if not is_numpy else 'cpu'

    # Convert all inputs to tensors (use float64 for better precision)
    tensors = [torch.as_tensor(v, dtype=torch.float64, device=device)
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
    w = torch.zeros((Qn, Y, X), device=device, dtype=torch.float64)
    
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
        F = torch.zeros((3, Y, X), dtype=torch.float64, device=device)
        F[0, :, :] = f_sum - 2 * E  # Mass conservation
        F[1, :, :] = f_ex - 2 * ux * H  # X-momentum conservation  
        F[2, :, :] = f_ey - 2 * uy * H  # Y-momentum conservation

        # Construct Jacobian matrix
        J = torch.zeros((3, 3, Y, X), dtype=torch.float64, device=device)
        
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
        if allow_early_termination:
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
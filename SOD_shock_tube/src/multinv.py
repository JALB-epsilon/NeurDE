import numpy as np
import torch

def _multinv_torch(M):
    """Ultra-fast 3x3 matrix inverse for torch.compile"""
    # Minimal regularization
    reg_val = 1e-12
    m = M.shape[-1]
    eye = torch.eye(m, dtype=M.dtype, device=M.device)
    
    if M.dim() > 2:
        eye = eye.expand_as(M)
    
    M_reg = M + reg_val * eye
    return torch.linalg.solve(M_reg, eye)

'''# Patch _multinv_torch to use the fast 3x3 version if possible
def _multinv_torch(M, device=None):
    """Matrix inverse function with proper batching support"""
    is_numpy = isinstance(M, np.ndarray)
    if is_numpy:
        M_torch = torch.as_tensor(M, dtype=torch.float64, device=device)
    else:
        M_torch = M.to(device=device, dtype=torch.float64)
    
    original_shape = M_torch.shape
    
    if len(original_shape) < 2:
        raise ValueError('M must have at least 2 dimensions')
    
    m, n = original_shape[-2], original_shape[-1]
    if m != n:
        raise ValueError('The last two dimensions of M must be square (m x m).')
    
    # Add small regularization for numerical stability
    reg_term = 1e-12 * torch.eye(m, dtype=M_torch.dtype, device=M_torch.device)
    if len(original_shape) > 2:
        batch_shape = original_shape[:-2]
        reg_term = reg_term.expand(*batch_shape, m, m)
    
    M_reg = M_torch + reg_term
    
    # Create identity matrices
    I = torch.eye(m, dtype=M_torch.dtype, device=M_torch.device) #* M_torch.norm() #multiply by norm for stability not full precision add operator norm
    #compute factorization Cholesky factorization 1e-15 regu

    if len(original_shape) > 2:
        batch_shape = original_shape[:-2]
        I = I.expand(*batch_shape, m, m)
    
    # Solve for inverse
    M_inv = torch.linalg.solve(M_reg, I)

    return M_inv.cpu().numpy() if is_numpy else M_inv'''
'''LU, pivots = torch.linalg.lu_factor(M_torch)
    num_iters = 4
    x = 0
    for _ in range(num_iters):
    r = b - M_torch @ x
    delta_x = torch.linalg.lu_solve(LU, pivots, r)
    x = x + delta_x'''
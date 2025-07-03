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


'''
# Patch _multinv_torch to use the fast 3x3 version if possible
def _multinv_torch(M, device=None):
    """Matrix inverse function with proper batching support"""
    is_numpy = isinstance(M, np.ndarray)
    
    # Use the current default dtype instead of hardcoded float64
    target_dtype = torch.get_default_dtype()
    
    if is_numpy:
        M_torch = torch.as_tensor(M, dtype=target_dtype, device=device)
    else:
        M_torch = M.to(device=device, dtype=target_dtype)
    
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


'''from scipy import sparse

def multinv(M):
    sn = M.shape
    m = sn[0]
    n = sn[1]
    if m != n:
        raise ValueError('The first two dimensions of M must be m x m slices.')
    p = np.prod(sn[2:])
    M = np.reshape(M, (m, n, p), order='F')

    # Build sparse matrix and solve
    I = np.reshape(np.arange(1, m*p+1), (m, 1, p), order='F')
    I = np.tile(I, (1, n, 1))  # m x n x p
    J = np.reshape(np.arange(1, n*p+1), (1, n, p), order='F')
    J = np.tile(J, (m, 1, 1))  # m x n x p
    M_sparse = sparse.coo_matrix((M.flatten('F'), (I.flatten('F')-1, J.flatten('F')-1)))
    M_sparse = M_sparse.tocsc() # Convert to CSC format

    RHS = np.tile(np.eye(m), (p, 1))
    X = sparse.linalg.spsolve(M_sparse, RHS)
    X = np.reshape(X, (n, p, m), order='F')
    X = np.transpose(X, (0, 2, 1))
    X = np.reshape(X, (n, m) + sn[2:], order='F')

    return X'''

'''import numpy as np
from scipy import sparse


def multinv(M):
    sn = M.shape
    m = sn[0]
    n = sn[1]
    if m != n:
        raise ValueError('The first two dimensions of M must be m x m slices.')
    p = np.prod(sn[2:])
    M = np.reshape(M, (m, n, p), order='F')

    # Build sparse matrix and solve
    I = np.reshape(np.arange(1, m*p+1), (m, 1, p), order='F')
    I = np.tile(I, (1, n, 1))  # m x n x p
    J = np.reshape(np.arange(1, n*p+1), (1, n, p), order='F')
    J = np.tile(J, (m, 1, 1))  # m x n x p
    M = sparse.coo_matrix((M.flatten('F'), (I.flatten('F')-1, J.flatten('F')-1)))

    RHS = np.tile(np.eye(m), (p, 1))
    X = sparse.linalg.spsolve(M, RHS)
    X = np.reshape(X, (n, p, m), order='F')
    X = np.transpose(X, (0, 2, 1))
    X = np.reshape(X, (n, m) + sn[2:], order='F')

    return X'''
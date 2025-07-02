import numpy as np
import torch

def _multinv_torch(M, device=None):
    is_numpy = isinstance(M, np.ndarray)
    if is_numpy:
        M_torch = torch.as_tensor(M, dtype=torch.float32, device=device)
    else:
        M_torch = M.to(device=device, dtype=torch.float32)

    original_shape = M_torch.shape
    m, n = original_shape[0], original_shape[1]
    if m != n:
        raise ValueError('The first two dimensions of M must be m x m slices.')
    permute_dims = list(range(2, len(original_shape))) + [0, 1]
    M_permuted = M_torch.permute(*permute_dims)

    # Create a batch of identity matrices with the same batch dimensions as M_permuted.
    batch_shape = M_permuted.shape[:-2]
    I = torch.eye(m, dtype=M_permuted.dtype, device=M_permuted.device)
    I_batch = I.expand(*batch_shape, m, m)
    # Use the highly optimized, batched version of linalg.solve to find the inverse.
    # This avoids any slow Python for-loops and is equivalent to linalg.inv.
    M_inv_permuted = torch.linalg.solve(M_permuted, I_batch)
    inv_permute_dims = [len(original_shape) - 2, len(original_shape) - 1] + list(range(len(original_shape) - 2))
    M_inv = M_inv_permuted.permute(*inv_permute_dims)
    return M_inv.cpu().numpy() if is_numpy else M_inv


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
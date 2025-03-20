import numpy as np
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
    M_sparse = sparse.coo_matrix((M.flatten('F'), (I.flatten('F')-1, J.flatten('F')-1)))
    M_sparse = M_sparse.tocsc() # Convert to CSC format

    RHS = np.tile(np.eye(m), (p, 1))
    X = sparse.linalg.spsolve(M_sparse, RHS)
    X = np.reshape(X, (n, p, m), order='F')
    X = np.transpose(X, (0, 2, 1))
    X = np.reshape(X, (n, m) + sn[2:], order='F')

    return X

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
 
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.io import loadmat
from scipy.sparse import coo_matrix

def multinv(M):
    # Check if the input is a square matrix for the first two dimensions
    sn = M.shape
    m, n = sn[0],sn[1]
    if m != n:
        raise ValueError('The first two dimensions of M must be m x m slices.')

    # Handle additional dimensions by reshaping
    p = np.prod(M.shape[2:])
    M = M.reshape((int(m), int(n), int(p)),order="F")
    # print(M.shape)
    # Build sparse matrix
    # Correctly generate index arrays for sparse matrix construction
    I = np.reshape(np.arange(0,m*p),(m,1,p),order ="F")
    # print(I.shape)
    I = np.tile(I,(1,n,1))
    J = np.reshape(np.arange(0,n*p),(1,n,p),order = "F")
    # print("after repeat",I.shape)
    # print(J.shape)
    J = np.tile(J,(m,1,1))


    # Assuming I, J, and M are numpy arrays of shape (3, 3, 15000)
    # Flatten the arrays
    ii = I.flatten()
    jj = J.flatten()
    mm = M.flatten()
    # Create the sparse matrix in COO format
    sparse_matrix = coo_matrix((mm, (ii, jj)))

    # Convert to CSR format for better performance in arithmetic operations
    sparse_matrix_csr = sparse_matrix.tocsr()
    
    # Prepare RHS as repeated identity matrices
    RHS = np.tile(np.eye(m), (int(p), 1))
    # print(sparse_matrix_csr.shape)
    # Solve the system
    X = spsolve(sparse_matrix_csr, RHS)
    
    # Reshape the result back to the original dimensions with the inverse for each slice
    X = np.reshape(X,(int(n),int(p),int(m)),order= "F")
    X = X.transpose(0,2,1)
    X = np.reshape(X,((n,m)+sn[2:]),order ="F")
    return X


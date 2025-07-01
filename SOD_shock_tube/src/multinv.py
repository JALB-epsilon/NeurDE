import numpy as np
import torch
'''from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
# from scipy.io import loadmat # Unused import
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
    return X'''

def _multinv_torch_3x3(M, device=None):
    """
    Fast closed-form batched inverse for 3x3 matrices using PyTorch.
    M: (..., 3, 3) tensor
    Returns: (..., 3, 3) tensor of inverses
    """
    if device is not None:
        M = M.to(device)
    # Assume M shape (..., 3, 3)
    a = M[..., 0, 0]
    b = M[..., 0, 1]
    c = M[..., 0, 2]
    d = M[..., 1, 0]
    e = M[..., 1, 1]
    f = M[..., 1, 2]
    g = M[..., 2, 0]
    h = M[..., 2, 1]
    i = M[..., 2, 2]
    
    A =   e * i - f * h
    B = -(d * i - f * g)
    C =   d * h - e * g
    D = -(b * i - c * h)
    E =   a * i - c * g
    F = -(a * h - b * g)
    G =   b * f - c * e
    H = -(a * f - c * d)
    I =   a * e - b * d
    
    det = a * A + b * B + c * C
    # Avoid division by zero
    det_safe = torch.where(torch.abs(det) < 1e-12, torch.ones_like(det), det)
    inv = torch.stack([
        torch.stack([A, D, G], dim=-1),
        torch.stack([B, E, H], dim=-1),
        torch.stack([C, F, I], dim=-1)
    ], dim=-2) / det_safe.unsqueeze(-1).unsqueeze(-1)
    # Set output to zero where det is zero (singular)
    inv = torch.where(torch.abs(det).unsqueeze(-1).unsqueeze(-1) < 1e-12, torch.zeros_like(inv), inv)
    return inv

# Patch _multinv_torch to use the fast 3x3 version if possible
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

    # Fast path for 3x3
    if m == 3 and n == 3:
        # Move batch to end: (3,3,Y,X) -> (Y,X,3,3)
        batch_shape = original_shape[2:]
        M_batched = M_torch.permute(*range(2, len(original_shape)), 0, 1)
        M_inv_batched = _multinv_torch_3x3(M_batched, device=device)
        # Move back to (3,3,Y,X)
        M_inv = M_inv_batched.permute(-2, -1, *range(0, len(batch_shape)))
        return M_inv.cpu().numpy() if is_numpy else M_inv

    # Generic path for other sizes
    permute_dims = list(range(2, len(original_shape))) + [0, 1]
    M_permuted = M_torch.permute(*permute_dims)
    M_inv_permuted = torch.linalg.inv(M_permuted)
    inv_permute_dims = [len(original_shape) - 2, len(original_shape) - 1] + list(range(len(original_shape) - 2))
    M_inv = M_inv_permuted.permute(*inv_permute_dims)
    return M_inv.cpu().numpy() if is_numpy else M_inv

'''def _multinv_torch_sparse(M, device=None):
    """
    Calculates the inverse of a batch of small matrices using PyTorch's
    sparse linear solver, replicating the logic from the original `multinv`.

    NOTE: `torch.sparse.linalg.spsolve` is currently CPU-only and does not
    support autograd. This function is therefore not suitable for training
    loops where gradients are required. It is provided for completeness and
    to match the original implementation's strategy.

    Args:
        M (np.ndarray or torch.Tensor): A NumPy array or Torch tensor of shape
                                        (m, n, ...), where each (m, n) slice
                                        is a matrix to be inverted. 'm' must
                                        be equal to 'n'.
        device (str, optional): The target device for the output, e.g., 'cpu'
                                or 'cuda'. If M is a tensor, its device is
                                used and this argument is ignored. If M is a
                                NumPy array, this argument specifies the
                                output device (defaults to 'cpu').

    Returns:
        np.ndarray or torch.Tensor: An array/tensor with the same shape as M,
                                    containing the inverses. The return type
                                    matches the input type.
    """
    is_numpy = isinstance(M, np.ndarray)

    # Determine the target device for the final output.
    if is_numpy:
        output_device = device if device is not None else 'cpu'
    else:
        output_device = M.device

    # The sparse solver is CPU-only. All solver-related tensors must be on the CPU.
    solver_device = 'cpu'
    if output_device != solver_device:
        print(f"Warning: torch.sparse.linalg.spsolve is CPU-only. Computation will be on {solver_device}, and the result moved to {output_device}.")

    # Ensure input tensor is on the CPU for the solver.
    if is_numpy:
        M_torch = torch.as_tensor(M, dtype=torch.float32, device=solver_device)
    else:
        M_torch = M.to(device=solver_device, dtype=torch.float32)

    original_shape = M_torch.shape
    m, n = original_shape[0], original_shape[1]
    if m != n:
        raise ValueError('The first two dimensions of M must be m x m slices.')

    batch_dims = torch.Size(original_shape[2:])
    p = batch_dims.numel()

    # Permute to (..., m, n) and then flatten the batch dimensions to (p, m, n)
    permute_dims = list(range(2, len(original_shape))) + [0, 1]
    M_permuted = M_torch.permute(*permute_dims)
    values = M_permuted.flatten()

    # Generate sparse indices for the block-diagonal matrix on the solver_device
    row_ids = torch.arange(m, device=solver_device).view(1, m, 1).expand(p, m, n)
    col_ids = torch.arange(n, device=solver_device).view(1, 1, n).expand(p, m, n)
    offsets = (torch.arange(p, device=solver_device) * m).view(p, 1, 1)

    final_rows = (row_ids + offsets).flatten()
    final_cols = (col_ids + offsets).flatten()
    indices = torch.stack([final_rows, final_cols])

    # Create the large sparse block-diagonal matrix
    sparse_M = torch.sparse_coo_tensor(indices, values, size=(m * p, n * p), device=solver_device)

    # Prepare RHS (p stacked identity matrices)
    RHS = torch.eye(m, device=solver_device).repeat(p, 1)

    # Solve the system
    X_solved = torch.sparse.linalg.spsolve(sparse_M, RHS)

    # Reshape result back to original format: (m, n, ...)
    X_bmm = X_solved.view(p, m, m)
    X_unflattened = X_bmm.view(*batch_dims, m, n)
    permute_back_dims = [len(batch_dims), len(batch_dims) + 1] + list(range(len(batch_dims)))
    X_final_cpu = X_unflattened.permute(*permute_back_dims)

    # Move the final result to the originally requested device
    X_final = X_final_cpu.to(output_device)

    return X_final.cpu().numpy() if is_numpy else X_final
'''
import torch

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    
    Args:
        A: A symmetric positive-definite matrix (2D list or torch.Tensor)
    
    Returns:
        L: Lower triangular torch.Tensor such that A = L @ L.T,
           or -1 if decomposition is not possible
    """
    # Your code here
    try: 
        return torch.linalg.cholesky(A)
    except (ValueError, TypeError, RuntimeError):
        return -1
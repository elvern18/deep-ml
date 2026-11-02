import torch

def qr_decomposition(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform QR decomposition using Gram-Schmidt process.
    
    Args:
        A: An m x n matrix as torch.Tensor
    
    Returns:
        Tuple of (Q, R) where Q is orthogonal and R is upper triangular
    """
    Q, R = torch.linalg.qr(A)

    signs = torch.sign(torch.diag(R)) 
    signs[signs == 0] = 1
    Q = Q * signs.unsqueeze(0)
    R = signs.unsqueeze(1) * R 
    return Q, R 
import torch

def linear_regression_normal_equation(X, y) -> torch.Tensor:
    """
    Solve linear regression via the normal equation using PyTorch.
    X: Tensor or convertible of shape (m,n); y: shape (m,) or (m,1).
    Returns a 1-D tensor of length n, rounded to 4 decimals.
    """
    X_t = torch.as_tensor(X, dtype=torch.float)
    y_t = torch.as_tensor(y, dtype=torch.float).reshape(-1,1)
    # Your implementation here
    
    # L = (Xw - Y)^2 
    # dL/dw = 2 X^T (Xw  -Y ) = 2 X^T * Xw - 2 X^T  * Y = 0 => w = (X^T * X)^-1  * X^ T * Y


    X_transposed = X_t.T

    w = torch.linalg.inv(X_transposed @ X_t) @ X_transposed @ y_t

    return w
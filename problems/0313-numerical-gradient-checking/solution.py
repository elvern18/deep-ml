import torch
from typing import Callable, Tuple

def numerical_gradient_check(f: Callable, x: torch.Tensor, analytical_grad: torch.Tensor, epsilon: float = 1e-7) -> Tuple[torch.Tensor, float]:
    """
    Perform numerical gradient checking using centered finite differences.

    Args:
        f: A function that takes a torch.Tensor and returns a scalar
        x: torch.Tensor, the point at which to check gradient
        analytical_grad: torch.Tensor, the analytically computed gradient
        epsilon: float, small value for finite difference approximation

    Returns:
        tuple: (numerical_grad, relative_error)
    """
    numerical_grad = torch.zeros_like(x)

    for i in range(x.numel()):

        x_plus = x.clone()
        x_minus = x.clone()

        x_plus.view(-1)[i] += epsilon
        x_minus.view(-1)[i] -= epsilon

        f_plus = f(x_plus)
        f_minus = f(x_minus)

        numerical_grad.view(-1)[i] = (f_plus - f_minus) / (2 * epsilon)

    # Calculate relative error
    numerator = torch.norm(analytical_grad - numerical_grad)
    denominator = torch.norm(analytical_grad) + torch.norm(numerical_grad)

    relative_error = (numerator / denominator).item()

    return numerical_grad, relative_error

import torch

def activation_derivatives(x: float) -> dict[str, float]:
    results = {}

    for name, activation in [
        ("sigmoid", torch.sigmoid),
        ("tanh", torch.tanh),
        ("relu", torch.relu)
    ]:
        # Create tensor with gradient tracking enabled
        t = torch.tensor(x, dtype=torch.float32, requires_grad=True)

        # Forward pass
        y = activation(t)

        # Backward pass
        y.backward()

        # Extract derivative
        results[name] = t.grad.item()

    return results

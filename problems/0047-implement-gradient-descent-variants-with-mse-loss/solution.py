import torch

def gradient_descent(
    X: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    learning_rate: float,
    n_epochs: int,
    batch_size: int = 1,
    method: str = "batch"
) -> torch.Tensor:

    m = X.shape[0]
    weights = weights.clone().detach().requires_grad_(True)

    for _ in range(n_epochs):

        if method == "batch":
            # One update using all samples
            X_batch = X
            y_batch = y

            predictions = X_batch @ weights
            loss = torch.mean((predictions - y_batch) ** 2)

            loss.backward()

            with torch.no_grad():
                weights -= learning_rate * weights.grad

            weights.grad.zero_()

        elif method == "stochastic":
            # One update for each sample, in original order
            for i in range(m):

                X_batch = X[i:i+1]
                y_batch = y[i:i+1]

                predictions = X_batch @ weights
                loss = torch.mean((predictions - y_batch) ** 2)

                loss.backward()

                with torch.no_grad():
                    weights -= learning_rate * weights.grad

                weights.grad.zero_()

        elif method == "mini_batch":
            # Consecutive, non-overlapping batches
            for start in range(0, m, batch_size):

                end = min(start + batch_size, m)

                X_batch = X[start:end]
                y_batch = y[start:end]

                predictions = X_batch @ weights
                loss = torch.mean((predictions - y_batch) ** 2)

                loss.backward()

                with torch.no_grad():
                    weights -= learning_rate * weights.grad

                weights.grad.zero_()

        else:
            raise ValueError(
                "method must be 'batch', 'stochastic', or 'mini_batch'"
            )

    return weights
    

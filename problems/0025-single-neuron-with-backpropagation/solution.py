
import torch
import torch.nn as nn

def train_neuron(
    features: torch.Tensor,
    labels: torch.Tensor,
    initial_weights: torch.Tensor,
    initial_bias: float,
    learning_rate: float,
    epochs: int
) -> tuple[list[float], float, list[float]]:

    # Initialize trainable parameters
    weights = initial_weights.clone().detach().float().requires_grad_(True)
    bias = torch.tensor(float(initial_bias), requires_grad=True)

    # Define optimizer and loss function
    optimizer = torch.optim.SGD([weights, bias], lr=learning_rate)
    loss_fn = nn.MSELoss()

    mse_values = []

    for epoch in range(epochs):
        # 1. Reset gradients
        optimizer.zero_grad()

        # 2. Forward pass
        z = torch.matmul(features, weights) + bias
        predictions = torch.sigmoid(z)

        # 3. Calculate MSE loss
        loss = loss_fn(predictions, labels.float())
        mse_values.append(round(loss.item(), 4))

        # 4. Backpropagation
        loss.backward()

        # 5. Update weights and bias
        optimizer.step()

    return (
        [round(w, 4) for w in weights.detach().tolist()],
        round(bias.item(), 4),
        mse_values
    )

import numpy as np

def dense_backward(a_prev: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:

    # Forward pass
    z = W @ a_prev + b
    pred = 1 / (1 + np.exp(-z))  # Sigmoid

    # Backward pass
    dz = (pred - y) * pred * (1 - pred) * 2

    dW = np.outer(dz, a_prev)
    db = dz
    da_prev = W.T @ dz

    return {
        "dW": dW.tolist(),
        "db": db.tolist(),
        "da_prev": da_prev.tolist()
    }

# import torch 
# def dense_backward(a_prev: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:
#     """
#     Compute gradients of the squared-error cost w.r.t. W, b, and a_prev
#     for a dense layer z = W @ a_prev + b followed by sigmoid activation.

#     Args:
#         a_prev: activations from previous layer, shape (K,)
#         W: weight matrix, shape (J, K)
#         b: bias vector, shape (J,)
#         y: target vector, shape (J,)

#     Returns:
#         dict with keys 'dW' (J x K nested list), 'db' (list of length J),
#         and 'da_prev' (list of length K).
#     """
#     # Your code here
#     w = torch.tensor(W, requires_grad = True)
#     bias = torch.tensor(b, requires_grad = True)
#     target = torch.tensor(y) 
#     x = torch.tensor(a_prev, requires_grad = True)

#     z = torch.matmul(w, x) + bias  
#     pred = torch.sigmoid(z) 
#     loss = torch.nn.functional.mse_loss(pred, target) 
#     loss.backward() 

#     return {
#         "dW": w.grad.tolist(),
#         "db": bias.grad.tolist(),
#         "da_prev": x.grad.tolist()
#     }

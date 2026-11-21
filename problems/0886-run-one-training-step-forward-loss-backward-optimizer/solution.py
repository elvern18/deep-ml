import torch
import torch.nn as nn
import torch.nn.functional as F

def train_one_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float) -> float:
    # TODO: build an SGD optimizer, run one full forward/loss/backward/step cycle,
    # and return the pre-update loss as a Python float.
    
   #  x_tensor = torch.tensor(x, require_grad = True ) 
    # y_tensor = torch.tensor(y, require_grad = True )

    opt = torch.optim.SGD(model.parameters(), lr = lr) 

    y_hat = model(x) 
    loss = F.mse_loss(y_hat, y)
    opt.zero_grad()
    loss.backward()
    opt.step() 

    return loss.item()

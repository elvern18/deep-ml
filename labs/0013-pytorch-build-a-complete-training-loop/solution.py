# --- cell 1 ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a PyTorch model and return training history.
    
    This is the standard PyTorch training pattern you'll use everywhere.
    Now you can use torch.optim to handle the gradient updates!
    
    Args:
        model: nn.Module to train
        X_train: training features, shape (N, ...)
        y_train: training labels, shape (N,)
        X_val: validation features, shape (M, ...)
        y_val: validation labels, shape (M,)
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate
    
    Returns:
        history: List of dicts, one per epoch, with keys:
            - 'epoch': epoch number (starting from 1)
            - 'train_loss': average training loss for the epoch
            - 'val_loss': validation loss after the epoch
            - 'val_accuracy': validation accuracy after the epoch
    
    Steps:
        1. Create optimizer: optim.Adam(model.parameters(), lr=lr)
        2. Create loss function: nn.CrossEntropyLoss()
        3. For each epoch:
            a. Shuffle training data
            b. Loop over mini-batches:
                - optimizer.zero_grad()
                - Forward pass
                - Compute loss
                - loss.backward()
                - optimizer.step()
            c. Compute validation accuracy
            d. Append metrics to history
        4. Return history
    
    Hints:
        - torch.randperm(n) gives a random permutation for shuffling
        - Use model.train() before training, model.eval() before validation
        - Use torch.no_grad() during validation
        - logits.argmax(dim=1) gives predicted classes
    """
    # TODO: Implement the training loop
    
    opt = torch.optim.Adam(model.parameters(), lr = lr) 
        
    
    history = []
    loss_fn = nn.CrossEntropyLoss() 
    N = X_train.shape[0]
            
    for epoch in range(1, 1 + epochs):
       model.train() 
       perm = torch.randperm(N)
       total_loss = 0 
       
       for start in range(0, N, batch_size): 
           i = perm[start: start + batch_size]
           xb, yb = X_train[i], y_train[i] 
           
           opt.zero_grad() 
           yb_hat = model(xb) 
           loss = loss_fn(yb_hat, yb)
           total_loss += loss.item() * len(i) 
           loss.backward() 
           opt.step() 
       
       train_loss = total_loss / N
       # ---- validate ---
       model.eval()
       with torch.no_grad():                        # c. no graph, no grads
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val).item()
            val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()
            
            
       history.append({                             # d. record metrics
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        

    
    return history
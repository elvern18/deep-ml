import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # stable log-softmax: subtract the row max so every exp argument is <= 0
    m = logits.max(dim=-1, keepdim=True).values               # (N, 1)
    shifted = logits - m                                       # (N, C)
    log_norm = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))   # (N, 1)
    log_probs = shifted - log_norm                             # (N, C)

    # log-prob of the correct class for each row
    picked = log_probs.gather(dim=1, index=targets.long().unsqueeze(1)).squeeze(1)   # (N,)

    return -picked.mean()
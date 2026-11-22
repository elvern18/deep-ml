import torch 
def clip_gradients_by_global_norm(gradients: list[torch.Tensor], max_norm: float) -> list[torch.Tensor]:

    # L2 norm across all tensors, as if they were one vector
    global_norm = torch.sqrt(sum(torch.sum(g.float() ** 2) for g in gradients))

    if global_norm >= max_norm:
        scale = max_norm / global_norm  
        return [g * scale for g in gradients]

    return gradients
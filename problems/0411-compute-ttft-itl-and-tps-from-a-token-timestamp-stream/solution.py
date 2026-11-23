import torch

def compute_inference_metrics(timestamps: torch.Tensor) -> dict:
    """
    Compute LLM inference performance metrics from token timestamps.
    
    Args:
        timestamps: 1D torch.Tensor where timestamps[0] is the request start time
                    and timestamps[1:] are the times when each output token was generated.
    
    Returns:
        Dictionary with keys 'ttft', 'tps', 'itl' containing the metric values as scalar tensors.
    """
    
    ttft = timestamps[1] - timestamps[0]

    total_time = timestamps[-1] - timestamps[0]
    tps = (len(timestamps) - 1) / total_time

    itl = torch.tensor(0)
    if len(timestamps) != 2:
        itl = (timestamps[-1] - timestamps[1]) / (len(timestamps) - 2)

    return {
        "ttft": ttft, 
        "tps": tps,
        "itl": itl,
    }


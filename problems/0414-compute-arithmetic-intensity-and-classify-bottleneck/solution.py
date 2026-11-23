def compute_arithmetic_intensity(flops: float, bytes_accessed: float, peak_performance: float, peak_bandwidth: float) -> dict:
    """
    Analyze a computational kernel using the Roofline Model.
    
    Args:
        flops: Total floating-point operations of the kernel
        bytes_accessed: Total bytes transferred to/from memory
        peak_performance: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)
    
    Returns:
        Dictionary with arithmetic_intensity, ridge_point, bottleneck,
        achieved_performance, and utilization_percent
    """
    
    ai = flops / bytes_accessed
    ridge_point = peak_performance / peak_bandwidth

    t_compute = flops / peak_performance
    t_memory = bytes_accessed / peak_bandwidth

    bottleneck = "" 
    if t_compute >= t_memory:
        bottleneck = "compute-bound"
    else:
        bottleneck = "memory-bound"
    
    achieved_performance = flops / max(t_memory, t_compute)
    utilization_percent = achieved_performance / peak_performance * 100 

    return {
        "arithmetic_intensity": ai,
        "ridge_point": ridge_point,
        "bottleneck": bottleneck, 
        "achieved_performance": achieved_performance,
        "utilization_percent": utilization_percent
    }


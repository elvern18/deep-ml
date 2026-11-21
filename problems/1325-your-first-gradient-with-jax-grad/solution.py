import jax
import jax.numpy as jnp

def df_dx(x):
    """Derivative of f(x) = x**3 + 2x at float x, via jax.grad. Returns float."""
    
    f = lambda y: y**3 + 2*y
    grad_func = jax.grad(f)
    return grad_func(x) 
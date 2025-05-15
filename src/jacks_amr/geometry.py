import jax.numpy as jnp

def normal_vector(dim, direction, n_dims):
    result = [0.0]*n_dims
    result[dim] = -1.0 if direction == 'left' else 1.0
    return jnp.array(result)

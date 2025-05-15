import jax.numpy as jnp


def first_non_nan(a, b):
    return jnp.where(jnp.isnan(a), b, a)

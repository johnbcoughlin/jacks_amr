import jax.numpy as jnp
import jax

def coarsen_by_2(arr, n_dims):
    if n_dims == 1:
        return 0.5 * (arr[0::2] + arr[1::2])
    elif n_dims == 2:
        return 0.25 * (arr[0::2, 0::2] + arr[1::2, 0::2] + arr[0::2, 1::2] + arr[1::2, 1::2])
    else:
        raise ValueError()


def prolong_by_2(arr, n_dims):
    new_shape = jax.tree.map(lambda s: s*2, arr.shape)
    
    if n_dims == 1:
        return jnp.zeros(new_shape) \
            .at[0::2].set(arr) \
            .at[1::2].set(arr)
    elif n_dims == 2:
        return jnp.zeros(new_shape) \
            .at[0::2, 0::2].set(arr) \
            .at[1::2, 0::2].set(arr) \
            .at[0::2, 1::2].set(arr) \
            .at[1::2, 1::2].set(arr)
    else:
        raise ValueError()


def first_non_nan(a, b):
    return jnp.where(jnp.isnan(a), b, a)

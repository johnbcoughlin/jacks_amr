import jax.numpy as jnp

def coarsen_face_values_by_2(fine_arr, dim, n_dims):
    if n_dims == 1:
        return fine_arr[0:-1:2]
        
    if n_dims == 2 and dim == 0:
        return fine_arr[0:-1:2, 0::2] + fine_arr[0:-1:2, 1::2]
        
    if n_dims == 2 and dim == 1:
        return fine_arr[0::2, 0:-1:2] + fine_arr[1::2, 0:-1:2]
        
    if n_dims == 3:
        if dim == 0:
            return fine_arr[0:-1:2, 0::2, 0::2] + fine_arr[0:-1:2, 0::2, 1::2] + \
            fine_arr[0:-1:2, 1::2, 0::2] + fine_arr[0:-1:2, 1::2, 1::2]
            
        if dim == 1:
            return fine_arr[0::2, 0:-1:2, 0::2] + fine_arr[0::2, 0:-1:2, 1::2] + \
            fine_arr[1::2, 0:-1:2, 0::2] + fine_arr[1::2, 0:-1:2, 1::2]
            
        if dim == 2:
            return fine_arr[0::2, 0::2, 0:-1:2] + fine_arr[0::2, 1::2, 0:-1:2] + \
            fine_arr[1::2, 0::2, 0:-1:2] + fine_arr[1::2, 1::2, 0:-1:2]
            
            
def first_non_nan(a, b):
    return jnp.where(jnp.isnan(a), b, a)
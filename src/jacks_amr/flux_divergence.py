from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx

from .amr import AMRGridFunction
from .face_reductions import reduce_face_integrals

def flux_differences_for_divergence(face_fluxes, n_dims):
    if n_dims == 1:
        return face_fluxes[0][:, 1:] - face_fluxes[0][:, :-1]

    if n_dims == 2:
        return (face_fluxes[0][:, 1:, :] - face_fluxes[0][:, :-1, :]) + \
            (face_fluxes[1][:, :, 1:] - face_fluxes[1][:, :, :-1])


@partial(jax.jit, static_argnums=(1, 2))
def flux_divergence(q, F_hat, bcs):
    '''
    params:
        - q: The AMRGridFunction of unknowns
        - F_hat: The numerical flux. A callable (q_in, q_out, n) -> n cdot flux
    '''
    grid = q.grid
    n_dims = grid.n_dims
    n_levels = len(grid.levels)
    
    # Using a single-point quadrature...
    def numerical_flux_face_integral(q_in, q_out, n, face_area):
        numerical_flux = F_hat(q_in, q_out, n)
        return numerical_flux * face_area

    final_face_fluxes = reduce_face_integrals(q, numerical_flux_face_integral, bcs)

    div_F = [jnp.zeros_like(q.level_values[i]) for i in range(n_levels)]
    for i in range(n_levels):
        div_F = eqx.tree_at(
            where=lambda t: t[i],
            pytree=div_F,
            replace_fn=lambda t: t + flux_differences_for_divergence(final_face_fluxes[i], n_dims)
        )

    return AMRGridFunction(grid, div_F)
from .face_reductions import reduce_face_integrals
from .flux_divergence import flux_differences_for_divergence
import jax.numpy as jnp
import jax
import equinox as eqx


def approximate_gradient_indicator(q, bcs):
    grid = q.grid
    n_dims = grid.n_dims
    n_levels = len(grid.levels)
    L0_dx = grid.grid_factory.L0_dx
    vol = 1.0
    for dim in range(n_dims):
        vol *= L0_dx[dim]
    L0_dx_avg = vol ** (1/n_dims)
    
    def face_jump_integral(q_in, q_out, n, face_area):
        return (q_out - q_in) * face_area
        
    face_jump_integrals = reduce_face_integrals(q, face_jump_integral, bcs)    
    
    perimeters = [
        tuple([(L0_dx_avg / (2**level_idx)) ** (n_dims-1) for dim in range(n_dims)]) 
        for level_idx in range(n_levels)
    ]
    diameters = [
        tuple([(L0_dx_avg / (2**level_idx)) for dim in range(n_dims)]) 
        for level_idx in range(n_levels)
    ]
    
    directional_derivatives = jax.tree.map(
        lambda t, p, d: t / p / d,
        face_jump_integrals, perimeters, diameters)
    
    grad_q = [jnp.zeros((n_dims, *q.level_values[i].shape)) for i in range(n_levels)]
    for i in range(n_levels):
        grad_q = eqx.tree_at(
            where=lambda t: t[i],
            pytree=grad_q,
            replace_fn=lambda t: flux_differences_for_divergence(directional_derivatives[i], n_dims)
        )
        
    gradient_scale_lengths = jax.tree.map(
        lambda gq, q: jnp.abs(q) / jnp.abs(gq),
        grad_q, q.level_values)
    indicators = jax.tree.map(
        lambda gsl, d: d / gsl,
        gradient_scale_lengths, [diams[0] for diams in diameters])
        
    return indicators
    
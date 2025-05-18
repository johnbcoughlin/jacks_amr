import jax.numpy as jnp
import numpy as np
import jax
import equinox as eqx
from functools import partial

@partial(jax.jit, static_argnums=(1, 2))
def refine_grid_for_function(q, bcs, indicator, 
    top_fraction_of_cells=0.2, bottom_fraction_of_cells=0.4):
    grid = q.grid
    n_dims = grid.n_dims
    n_levels = len(grid.levels)
    
    # Compute refinement criterion for every cell at every level
    indicator_vals = indicator(q, bcs).level_values
    
    indicators_flat = jnp.concatenate(
        [v.flatten() for v in indicator_vals])
    indicators_sort = jnp.argsort(indicators_flat)
    indicators_sorted = indicators_flat[indicators_sort]
    
    
    cutoffs = jnp.array([bottom_fraction_of_cells, 1.0 - top_fraction_of_cells])
    cutoffs = jnp.nanquantile(indicators_sorted, cutoffs)
    refinement_cutoff = cutoffs[1]
    coarsen_cutoff = cutoffs[0]
    jax.debug.print("refinement_cutoff: {}", refinement_cutoff)
    jax.debug.print("coarsen_cutoff: {}", coarsen_cutoff)
    
    def should_coarsen_level(level_idx):
        below_cutoff = indicator_vals[level_idx] <= coarsen_cutoff
        return jax.vmap(lambda block: jnp.all(block))(below_cutoff)
        
    should_coarsen = jax.tree.map(should_coarsen_level, list(range(n_levels)))
    
    # Coarsen blocks first
    for fine_level_idx in range(1, n_levels):
        def coarsen_block(carry, fine_block_active_idx):
            fine_block_indices = grid.block_indices(fine_level_idx, fine_block_active_idx)
            func, new_grid = carry
            func = func.with_block_coarsened(fine_level_idx, fine_block_indices)
            new_grid = new_grid.with_block_inactive(fine_level_idx, fine_block_indices)
            func = eqx.tree_at(
                where=lambda t: t.grid,
                pytree=func,
                replace_fn=lambda t: new_grid)
            return (func, new_grid)
            
        def maybe_coarsen_block(carry, fine_block_active_idx):
            if fine_level_idx == n_levels-1:
                is_refined_at_next_level = False
            else:
                indexer = grid.index_contained_fine_level_blocks(fine_level_idx, 
                    grid.block_indices(fine_level_idx, fine_block_active_idx))
                is_refined_at_next_level = jnp.any(
                    grid.levels[fine_level_idx+1].block_index_map[indexer] != -1)
            
            can_refine = jnp.logical_not(is_refined_at_next_level)
                
            return jax.lax.cond(
                (fine_block_active_idx != -1) & (should_coarsen[fine_level_idx][fine_block_active_idx]) & can_refine,
                lambda: coarsen_block(carry, fine_block_active_idx),
                lambda: carry), None
            
        carry, _ = jax.lax.scan(maybe_coarsen_block, (q, grid), 
            jnp.arange(grid.level_specs[fine_level_idx].n_blocks))
        q, grid = carry
        
    # After coarsening, refine
    def should_consider_for_refinement_level(level_idx):
        above_cutoff = indicator_vals[level_idx] >= refinement_cutoff
        return jax.vmap(lambda block: jnp.any(block))(above_cutoff)
    
    should_consider_for_refinement = jax.tree.map(
        should_consider_for_refinement_level, list(range(n_levels)))
    
    for coarse_level_idx in range(n_levels-1):
        fine_level_idx = coarse_level_idx+1
        
        def refine_inside_coarse_block(carry, coarse_block_active_idx):
            coarse_block_indices = grid.block_indices(coarse_level_idx, coarse_block_active_idx)
            coarse_block_origin = grid.indices_to_origin(coarse_level_idx, coarse_block_indices)
            contained_fine_block_origins = jax.tree.map(
                lambda coarse_origin, coarse_block_size, fine_block_size: coarse_origin*2 + jnp.arange(0, coarse_block_size*2, fine_block_size),
                coarse_block_origin,
                grid.level_specs[coarse_level_idx].block_shape,
                grid.level_specs[fine_level_idx].block_shape)
            flattened_fine_block_origins = jax.tree.map(
                lambda a: a.flatten(),
                tuple(jnp.meshgrid(*contained_fine_block_origins, indexing='ij')))
            flattened_fine_block_indices = grid.origin_to_indices(fine_level_idx, flattened_fine_block_origins)
            
            # Evaluate the refinement criterion for the fine block and refine to it if necessary
            def refine_single_block(carry, fine_block_indices):
                func, new_grid = carry
                new_grid, new_active_idx = new_grid.with_block_active(fine_level_idx, fine_block_indices)
                func = func.with_block_refined(fine_level_idx, new_active_idx, fine_block_indices)
                func = eqx.tree_at(
                    where=lambda t: t.grid,
                    pytree=func,
                    replace_fn=lambda t: new_grid)
                return (func, new_grid)
                
            def maybe_refine_single_block(carry, fine_block_indices):
                fine_block_origin = grid.indices_to_origin(fine_level_idx, fine_block_indices)
                indexer = grid.index_fine_block_in_coarse(fine_level_idx, fine_block_indices, coarse_block_indices)
                coarse_indicators = indicator_vals[coarse_level_idx][indexer]
                _, new_grid = carry
                has_space = new_grid.levels[fine_level_idx].n_active < grid.level_specs[fine_level_idx].n_blocks
                carry = jax.lax.cond(jnp.any(coarse_indicators >= refinement_cutoff) & has_space,
                    lambda: refine_single_block(carry, fine_block_indices),
                    lambda: carry)
                #if coarse_level_idx == 0:
                    #jax.debug.print("condition: {}", jnp.any(coarse_indicators >= refinement_cutoff) & has_space)
                    #jax.debug.print("level 1: {}", carry[0].grid.levels[1].block_index_map)
                return carry, None
                
            carry, _ = jax.lax.scan(maybe_refine_single_block, carry, flattened_fine_block_indices)
            return carry, None
        
        carry, _ = jax.lax.scan(refine_inside_coarse_block, (q, grid), 
            jnp.arange(grid.level_specs[coarse_level_idx].n_blocks))
        q, grid = carry
        
    return q
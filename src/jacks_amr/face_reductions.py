from typing import Callable
import jax.numpy as jnp
import jax
import equinox as eqx

from .amr import AMRGridFunction
from .arrays import first_non_nan
from .geometry import normal_vector


def coarsen_face_values_by_2(fine_arr, dim, n_dims):
    """
    Coarsens an (2m+1) x 2m array of face values on level n+1 to a (m+1) x m array of
    face values on level n.
    """
    if n_dims == 1:
        return fine_arr[0::2]

    if n_dims == 2 and dim == 0:
        return fine_arr[0::2, 0::2] + fine_arr[0::2, 1::2]

    if n_dims == 2 and dim == 1:
        return fine_arr[0::2, 0::2] + fine_arr[1::2, 0::2]

    if n_dims == 3:
        if dim == 0:
            return fine_arr[0::2, 0::2, 0::2] + fine_arr[0::2, 0::2, 1::2] + \
            fine_arr[0::2, 1::2, 0::2] + fine_arr[0::2, 1::2, 1::2]

        if dim == 1:
            return fine_arr[0::2, 0::2, 0::2] + fine_arr[0::2, 0::2, 1::2] + \
            fine_arr[1::2, 0::2, 0::2] + fine_arr[1::2, 0::2, 1::2]

        if dim == 2:
            return fine_arr[0::2, 0::2, 0::2] + fine_arr[0::2, 1::2, 0::2] + \
            fine_arr[1::2, 0::2, 0::2] + fine_arr[1::2, 1::2, 0::2]


def exchange_crosslevel_values(crosslevel_values, level, dim, level_spec, n_dims):
    """
    Exchanges crosslevel values between neighboring blocks of the same level
    """
    if n_dims == 1:
        return exchange_crosslevel_values_1d(crosslevel_values, level, level_spec, dim)
    elif n_dims == 2:
        return exchange_crosslevel_values_2d(crosslevel_values, level, level_spec, dim)


def exchange_crosslevel_values_1d(crosslevel_values, level, level_spec, dim):
    assert dim == 0
    block_array_shape = level_spec.block_array_shape
    return do_exchange_crosslevel_values(crosslevel_values, level, dim,
        ((jnp.arange(block_array_shape[0]-1),), (1+jnp.arange(block_array_shape[0]-1),)), 
        ((0,), (-1,)))


def exchange_crosslevel_values_2d(crosslevel_values, level, level_spec, dim):
    block_array_shape = level_spec.block_array_shape
    block_shape = level_spec.block_shape
    if dim == 0:
        return do_exchange_crosslevel_values(crosslevel_values, level, 0,
            ((jnp.arange(block_array_shape[0]-1), jnp.arange(block_array_shape[1])),
             (1+jnp.arange(block_array_shape[0]-1), jnp.arange(block_array_shape[1]))),
            ((0, jnp.arange(block_shape[1])), (-1, jnp.arange(block_shape[1]))))
    elif dim == 1:
        return do_exchange_crosslevel_values(crosslevel_values, level, 1,
            ((jnp.arange(block_array_shape[0]), jnp.arange(block_array_shape[1]-1)),
             (jnp.arange(block_array_shape[0]), 1+jnp.arange(block_array_shape[1]-1))),
            ((jnp.arange(block_shape[0]), 0), (jnp.arange(block_shape[0]), -1)))


def do_exchange_crosslevel_values(crosslevel_values, level, dim,
    leftright_block_index_map_shifts, leftright_slices):

    leftshift, rightshift = leftright_block_index_map_shifts
    active_pairs = (level.block_index_map[jnp.ix_(*leftshift)] != -1) & \
        (level.block_index_map[jnp.ix_(*rightshift)] != -1)
    pair_left_active_indices = jnp.where(active_pairs,
        (level.block_index_map[jnp.ix_(*leftshift)]),
        -1).flatten()
    pair_right_active_indices = jnp.where(active_pairs,
        (level.block_index_map[jnp.ix_(*rightshift)]),
        -1).flatten()

    def coalesce_pair(carry, pair_active_indices):
        pair_left_active_index, pair_right_active_index = pair_active_indices
        # TODO: can we replace these two calls to tree_at with a call that modifies
        # both locations at once by returning a tuple from "where"?
        left_slice, right_slice = leftright_slices
        leftright_ixs = jnp.ix_(*map(jnp.atleast_1d, [pair_left_active_index, *right_slice]))
        rightleft_ixs = jnp.ix_(*map(jnp.atleast_1d, [pair_right_active_index, *left_slice]))

        carry = carry.at[leftright_ixs] \
                .set(first_non_nan(carry[leftright_ixs],
                                   carry[rightleft_ixs]))

        carry = carry.at[rightleft_ixs] \
                .set(first_non_nan(carry[leftright_ixs],
                                   carry[rightleft_ixs]))
        return carry

    def maybe_coalesce_pair(carry, pair_active_indices):
        carry = jax.lax.cond(pair_active_indices[0] != -1,
            lambda: coalesce_pair(carry, pair_active_indices),
            lambda: carry)

        return carry, None

    crosslevel_values, _ = jax.lax.scan(maybe_coalesce_pair, crosslevel_values,
        (pair_left_active_indices, pair_right_active_indices))
    return crosslevel_values


def reduce_face_integrals(q: AMRGridFunction, face_integral: Callable, bcs: Callable) -> list[jax.Array]:
    """
    params:
        q: a grid function over which to compute the facewise reduction
        face_integral: a function `(left_cells, right_cells, leftright_normal, face_area) -> scalar`
        bcs: a function `(coords, copyout_cell_values) -> ghost_cell_values`
        
    Computes the sum of face_quantity(q_left, q_right, n) for each face adjoining an element.
    Returns a grid function with the same shape as q.
    """
    grid = q.grid
    n_dims = grid.n_dims
    n_levels = len(grid.levels)
    L0_dx = grid.grid_factory.L0_dx
    face_areas = []
    for level in range(n_levels):
        by_dim = []
        for dim in range(n_dims):
            area = 1.0
            for other_dim in range(n_dims):
                if other_dim != dim:
                    area *= L0_dx[other_dim] / (2 ** level)
            by_dim.append(area)
        face_areas.append(by_dim)
    
    def faces_array_shape(cells_array_shape, dim):
        l = [*cells_array_shape]
        # The leading dimension is the block number
        l[dim+1] += 1
        return tuple(l)

    final_face_values = [
        tuple([jnp.nan * jnp.ones(faces_array_shape(q.level_values[i].shape, dim)) for dim in range(n_dims)])
        for i in range(n_levels)
    ]
    intralevel_face_values = [
        tuple([jnp.nan * jnp.ones(faces_array_shape(q.level_values[i].shape, dim)) for dim in range(n_dims)])
        for i in range(n_levels)
    ]
    crosslevel_face_values = [
        tuple([jnp.nan * jnp.ones(faces_array_shape(q.level_values[i].shape, dim)) for dim in range(n_dims)])
        for i in range(n_levels)
    ]

    def compute_blockwise_values(level_idx, block_active_idx, block_indices, dim):
        block_origin = jax.tree.map(lambda s, b: s * b,
                                    block_indices, grid.level_specs[level_idx].block_shape)
        left_ghost_cells = q.ghost_cells(level_idx, block_origin, dim, 'left', bcs)
        right_ghost_cells = q.ghost_cells(level_idx, block_origin, dim, 'right', bcs)
        interior_cells = q.level_values[level_idx][block_active_idx, ...]
        all_cells = jnp.concatenate([left_ghost_cells, interior_cells, right_ghost_cells],
                                    axis=dim)

        left_cells = jnp.concatenate([left_ghost_cells, interior_cells], axis=dim)
        right_cells = jnp.concatenate([interior_cells, right_ghost_cells], axis=dim)

        n = normal_vector(dim, 'right', n_dims)
        return face_integral(left_cells, right_cells, n, face_areas[level_idx][dim])
        
    for dim in range(n_dims):
        final_face_values = eqx.tree_at(
            where=lambda tree: tree[-1][dim],
            pytree=final_face_values,
            replace_fn=lambda _: jax.vmap(
                        lambda i, indices: compute_blockwise_values(n_levels-1, i, indices, dim)
                    )(jnp.arange(grid.level_specs[-1].n_blocks),
                                 grid.levels[-1].block_indices))

        for level_idx in range(n_levels-2, -1, -1):
            fine_level_idx = level_idx+1
            
            def transfer_face_values(crosslevel_values, fine_block_active_index):
                fine_block_indices = jax.tree.map(
                    lambda indices: indices[fine_block_active_index],
                    grid.levels[fine_level_idx].block_indices)
                fine_block_origin = jax.tree.map(
                    lambda idx, s: idx * s,
                    fine_block_indices, grid.level_specs[fine_level_idx].block_shape)

                coarse_block_origin = jax.tree.map(
                    lambda origin, coarse_block_shape: coarse_block_shape * (origin // 2 // coarse_block_shape),
                    fine_block_origin, grid.level_specs[level_idx].block_shape)
                coarse_block_indices = jax.tree.map(
                    lambda origin, s: origin // s,
                    coarse_block_origin, grid.level_specs[level_idx].block_shape)

                offset = jax.tree.map(
                    lambda coarse, fine: fine // 2 - coarse,
                    coarse_block_origin, fine_block_origin)
                coarse_block_slices = jax.tree.map(
                    lambda o, s: o + jnp.arange((s+1) // 2),
                    offset, grid.level_specs[fine_level_idx].face_block_shapes[dim])
    
                coarse_block_active_index = grid.levels[level_idx].block_index_map[coarse_block_indices]
                fine_face_values = final_face_values[fine_level_idx][dim][fine_block_active_index]
                coarsened_face_values = coarsen_face_values_by_2(fine_face_values, dim, n_dims)

                crosslevel_values = jax.lax.cond(fine_block_active_index >= grid.levels[fine_level_idx].n_active,
                    lambda: crosslevel_values,
                    lambda: crosslevel_values \
                        .at[jnp.ix_(jnp.array([coarse_block_active_index]), *coarse_block_slices)].set(coarsened_face_values))

                return crosslevel_values, None
                
                
            # Transfer fine face values to this level by aggregating
            crosslevel_face_values = eqx.tree_at(
                where=lambda tree: tree[level_idx][dim],
                pytree=crosslevel_face_values,
                replace_fn=lambda crosslevel_values: jax.lax.scan(transfer_face_values, crosslevel_values,
                    jnp.arange(grid.level_specs[fine_level_idx].n_blocks))[0])

            # Exchange crosslevel face values between neighboring blocks
            crosslevel_face_values = eqx.tree_at(
                where=lambda tree: tree[level_idx][dim],
                pytree=crosslevel_face_values,
                replace_fn=lambda t: exchange_crosslevel_values(t, grid.levels[level_idx],
                    dim, grid.level_specs[level_idx], n_dims))

            intralevel_face_values = eqx.tree_at(
                where=lambda tree: tree[level_idx][dim],
                pytree=intralevel_face_values,
                replace_fn=lambda _: jax.vmap(
                            lambda i, indices: compute_blockwise_values(level_idx, i, indices, dim)
                        )(jnp.arange(grid.level_specs[level_idx].n_blocks),
                                    grid.levels[level_idx].block_indices))

            def finalize_face_values(_):
                crosslevel_values = crosslevel_face_values[level_idx][dim]
                intralevel_values = intralevel_face_values[level_idx][dim]
                return jnp.where(jnp.isnan(crosslevel_values), intralevel_values, crosslevel_values)

            final_face_values = eqx.tree_at(
                where=lambda tree: tree[level_idx][dim],
                pytree=final_face_values,
                replace_fn=finalize_face_values)

    return final_face_values
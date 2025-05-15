from jax._src.numpy.lax_numpy import cross
import jax.numpy as jnp
from opt_einsum.paths import re
import equinox as eqx
import jax
from jaxtyping import Array, Bool, PyTree, PyTreeDef
import math
from functools import partial

class AMRLevelSpec(eqx.Module):
    n_blocks: int
    block_array_shape: PyTree
    block_shape: PyTree
    face_block_shapes: PyTree
    level_shape: PyTree

    def __init__(self, level, L0_shape, n_blocks, block_shape):
        level_shape = jax.tree.map(lambda s: s * (2**level), L0_shape)

        n_dims = len(block_shape)
        self.n_blocks = n_blocks
        self.block_array_shape = jax.tree.map(lambda s1, s2: int(s1 / s2), level_shape, block_shape)
        self.block_shape = block_shape

        face_block_shapes = []
        for dim in range(n_dims):
            L = list(block_shape)
            L[dim] = L[dim] + 1
            face_block_shapes.append(tuple(L))
        self.face_block_shapes = tuple(face_block_shapes)
        self.level_shape = level_shape


    def block_ghost_cell_relative_indices(self, dim, direction):
        '''
        Returns an array of indices of the request ghost cells, relative to the origin
        of the requesting block.
        '''
        n_dims = len(self.block_shape)
        if direction == 'left':
            rel = jnp.array([-1])
        elif direction == 'right':
            rel = jnp.array([self.block_shape[dim]])

        relative_indices = [jnp.arange(self.block_shape[i]) for i in range(n_dims)]
        relative_indices[dim] = rel
        return tuple(relative_indices)


    def block_copyout_cell_relative_indices(self, dim, direction):
        '''
        Returns an array of indices of the request ghost cells, relative to the origin
        of the requesting block.
        '''
        n_dims = len(self.block_shape)
        if direction == 'left':
            rel = jnp.array([0])
        elif direction == 'right':
            rel = jnp.array([self.block_shape[dim]-1])

        relative_indices = [jnp.arange(self.block_shape[i]) for i in range(n_dims)]
        relative_indices[dim] = rel
        return tuple(relative_indices)



class AMRLevel(eqx.Module):
    n_active: int
    # A (K x M x N x ...) array whose entries are either -1, indicating an inactive block,
    # or an integer in the range [0, n_active)
    block_index_map: jax.Array
    # An n-dim tuple of 1D arrays of length n_blocks.
    block_indices: PyTree

    def __init__(self, block_index_map, n_active, block_indices):
        self.n_active = n_active
        self.block_index_map = block_index_map
        self.block_indices = block_indices


    def with_block_active(self, block_index):
        return jax.lax.cond(self.block_index_map[block_index] >= 0,
                            lambda: self,
                            lambda: AMRLevel(
                                self.block_index_map.at[block_index].set(self.n_active),
                                self.n_active+1,
                                jax.tree.map(lambda a, idx: a.at[self.n_active].set(idx),
                                             self.block_indices,
                                             block_index)))


class AMRGridFactory():
    '''
    The "factory class" for an adaptively mesh refined grid.
    An instance of this class is "static", meaning that it doesn't contain any
    information about actual mesh refinements, it just defines the coarsest grid and
    the parameters for how the mesh is to be refined.
    '''
    def __init__(self,
                 n_levels, n_components, L0_shape, lower, upper,
                 level_specs):
        self.n_components = n_components
        self.n_levels = n_levels
        self.L0_shape = L0_shape
        self.lower = lower
        self.upper = upper
        self.L0_dx = jax.tree.map(lambda l, u, s: (u - l) / s, lower, upper, L0_shape)
        self.level_specs = level_specs

        self.n_dims = len(L0_shape)

        assert level_specs[0].n_blocks == 1
        assert level_specs[0].block_shape == L0_shape

        def make_coordinate_array(l, u, L0_dim, dim, i):
            level_size = L0_dim * (2**i)
            dx = (u - l) / level_size

            def expand_dims(points):
                if self.n_dims == 1:
                    return points
                elif self.n_dims == 2:
                    if dim == 0:
                        return jnp.expand_dims(points, axis=1)
                    else:
                        return jnp.expand_dims(points, axis=0)
                elif self.n_dims == 3:
                    if dim == 0:
                        return jnp.expand_dims(points, axis=(1, 2))
                    elif dim == 1:
                        return jnp.expand_dims(points, axis=(0, 2))
                    else:
                        return jnp.expand_dims(points, axis=(0, 1))

            points = jnp.linspace(l+dx/2, u-dx/2, level_size)
            d = {
                'center': jnp.linspace(l+dx/2, u-dx/2, level_size),
                'lower': jnp.linspace(l, u-dx, level_size),
                'upper': jnp.linspace(l+dx, u, level_size),
            }
            return jax.tree.map(expand_dims, d)


        self.level_coordinates_center = [jax.tree.map(
            lambda l, u, L0_dim, dim: make_coordinate_array(l, u, L0_dim, dim, i)['center'],
            lower, upper, L0_shape, tuple(range(self.n_dims)))
                                  for i in range(n_levels)]
        self.level_coordinates_lower = [jax.tree.map(
            lambda l, u, L0_dim, dim: make_coordinate_array(l, u, L0_dim, dim, i)['lower'],
            lower, upper, L0_shape, tuple(range(self.n_dims)))
                                  for i in range(n_levels)]


    def empty_amr_level(self, level):
        spec = self.level_specs[level]
        block_index_map = -1 * jnp.ones(spec.block_array_shape, int)
        block_indices = jax.tree.map(lambda dim: jnp.repeat(-1, spec.n_blocks),
                                     tuple(range(self.n_dims)))
        return AMRLevel(block_index_map,
                        0,
                        block_indices)


    def base_grid(self):
        '''
        Returns the AMRGrid with the zeroth level fully populated, and all other levels empty.
        '''
        levels = [self.empty_amr_level(i) for i in range(self.n_levels)]
        spec = self.level_specs[0]
        block_index_map = jnp.zeros(spec.block_array_shape, int)
        block_indices = jax.tree.map(
                lambda arr: arr.at[:].set(0),
                levels[0].block_indices)
        levels[0] = AMRLevel(block_index_map, 1, block_indices)
        return AMRGrid(self.n_dims, levels, self.level_specs, 
            self.level_coordinates_center, self)


    @partial(jax.jit, static_argnums=(0, 1, 2))
    def refine_to_approximate(self, f, refinement_criterion):
        grid = self.base_grid()
        one_to_ndim = tuple(range(self.n_dims))

        def refine_one_level(grid, level_idx):
            coarse = {
                "center_coords": self.level_coordinates_center[level_idx],
                "spec": self.level_specs[level_idx],
            }
            fine = {
                "center_coords": self.level_coordinates_center[level_idx+1],
                "spec": self.level_specs[level_idx+1],
            }

            def evaluate_criteria_in_coarse_block(coarse_block_indices):
                coarse_coords = jax.tree.map(
                        lambda idxs, coords, axis: jnp.take(coords, idxs, axis=axis),
                        coarse_block_indices, coarse["center_coords"], one_to_ndim)

                # Find all fine blocks that lie inside the current coarse block
                contained_fine_block_indices = jax.tree.map(
                        lambda coarse_idx, coarse_block_size, fine_block_size, dim: coarse_idx*2 + jnp.arange(0, coarse_block_size*2, fine_block_size),
                        coarse_block_indices, coarse["spec"].block_shape, fine["spec"].block_shape, one_to_ndim)
                flattened_fine_block_indices = jax.tree.map(
                        lambda a: a.flatten(),
                        tuple(jnp.meshgrid(*contained_fine_block_indices, indexing='ij')))

                # Evaluate the refinement criterion for the fine block and activate it if necessary.
                def single_block_criterion(fine_block_indices):
                    fine_coords = jax.tree.map(
                            lambda idx, fine_block_size, coords, axis: jnp.take(coords, idx + jnp.arange(fine_block_size), axis=axis),
                            fine_block_indices, fine["spec"].block_shape, fine["center_coords"], one_to_ndim)
                    comparable_coarse_coords = jax.tree.map(
                            lambda idx, fine_block_size, coords, axis: jnp.take(coords, idx // 2 + jnp.arange(fine_block_size // 2), axis=axis),
                            fine_block_indices, fine["spec"].block_shape, coarse["center_coords"], one_to_ndim)

                    fine_values = f(*fine_coords)
                    coarse_values = jnp.tile(f(*comparable_coarse_coords), [2]*self.n_dims)

                    return refinement_criterion(coarse_values, fine_values)


                fine_block_criteria = jax.vmap(single_block_criterion)(flattened_fine_block_indices)

                return fine_block_criteria, flattened_fine_block_indices

            fine_block_criteria, flattened_fine_block_indices = jax.vmap(evaluate_criteria_in_coarse_block)(grid.levels[level_idx].block_indices)
            filtered_criteria = jnp.where(jnp.expand_dims(grid.levels[level_idx].block_indices[0] == -1, axis=1),
                                          -jnp.inf,
                                          fine_block_criteria)
            sorted_idcs = jnp.argsort(filtered_criteria.flatten())

            def with_block_active(carry_grid, fine_block_indices):
                block_indices = jax.tree.map(
                        lambda s: s,
                        fine_block_indices)
                carry_grid = eqx.tree_at(where=lambda grid: grid.levels[level_idx+1],
                                   pytree=carry_grid,
                                   replace_fn=lambda level: level.with_block_active(block_indices))
                return carry_grid, None

            K = 10 * (2**level_idx)
            top_K_blocks = jax.tree.map(lambda a: a.flatten()[sorted_idcs[-K:]], flattened_fine_block_indices)
            refined_grid, _ = jax.lax.scan(with_block_active, grid, top_K_blocks)
            return refined_grid

        for level_idx in range(4):
            grid = refine_one_level(grid, level_idx)

        return grid



class AMRGrid(eqx.Module):
    n_dims: int = eqx.field(static=True)
    levels: list[AMRLevel]
    level_specs: list[AMRLevelSpec] = eqx.field(static=True)
    level_coordinates_center: list[PyTree]
    grid_factory: AMRGridFactory = eqx.field(static=True)

    def __init__(self, n_dims, levels, level_specs, level_coordinates_center, grid_factory):
        self.n_dims = n_dims
        self.levels = levels
        self.level_specs = level_specs
        self.level_coordinates_center = level_coordinates_center
        self.grid_factory = grid_factory


    @partial(jax.jit, static_argnums=(1,))
    def approximate(self, f):
        one_to_ndim = tuple(range(self.n_dims))

        def approximate_single_level(level_idx):
            level = self.levels[level_idx]
            spec = self.level_specs[level_idx]
            center_coords = self.level_coordinates_center[level_idx]

            def approximate_single_block(block_indices):
                block_origin = jax.tree.map(lambda block_size, idx: idx * block_size,
                                            spec.block_shape, block_indices)
                coords = jax.tree.map(
                        lambda origin, block_size, coords, axis: jnp.take(
                            coords, origin + jnp.arange(block_size), axis=axis),
                        block_origin, spec.block_shape, center_coords, one_to_ndim)
                return f(*coords)

            vmapped = jax.vmap(approximate_single_block)(level.block_indices)
            mask = jnp.expand_dims(level.block_indices[0] == -1, axis=(1, 2))
            filtered = jnp.where(mask, jnp.nan, vmapped)

            return filtered

        return AMRGridFunction(self,
                               [approximate_single_level(i) for i in range(len(self.levels))])



class AMRGridFunction(eqx.Module):
    grid: AMRGrid
    level_values: list[jax.Array]

    def __init__(self, grid, level_values):
        self.grid = grid
        self.level_values = level_values


    def ghost_cells(self, level_idx, block_origin, dim, direction, boundary_condition) -> jnp.ndarray:
        block_indices = jax.tree.map(lambda s, shape: s // shape,
                                     block_origin, self.grid.level_specs[level_idx].block_shape)
        if direction == 'left':
            return jax.lax.cond(block_indices[dim] == 0,
                                lambda: self.ghost_cells_boundary(level_idx, block_origin, dim, direction, boundary_condition),
                                lambda: self.ghost_cells_interior(level_idx, block_origin, dim, direction))
        else: # direction == 'right'
            return jax.lax.cond(block_indices[dim] == self.grid.level_specs[level_idx].block_array_shape[dim]-1,
                                lambda: self.ghost_cells_boundary(level_idx, block_origin, dim, direction, boundary_condition),
                                lambda: self.ghost_cells_interior(level_idx, block_origin, dim, direction))


    @partial(jax.jit, static_argnums=(1, 3, 4))
    def ghost_cells_interior(self, level_idx, block_origin, dim, direction):
        '''
        Returns the array of ghost cell values bordering the level-`level_idx` block
        with origin `block_origin`, in `direction` along `dim`.
        '''
        grid = self.grid
        level_spec = grid.level_specs[level_idx]
        ghost_cells_shape = [*level_spec.block_shape]
        ghost_cells_shape[dim] = 1
        result = jnp.zeros(ghost_cells_shape)

        fine_ghost_cell_relative_indices = level_spec.block_ghost_cell_relative_indices(dim, direction)
        fine_ghost_cell_indices = jax.tree.map(
                lambda a, b: a + b, block_origin, fine_ghost_cell_relative_indices)
        fine_ghost_cell_first_index = jax.tree.map(
                lambda a: a.flatten()[0], fine_ghost_cell_indices)

        for neighbor_level_idx in range(level_idx + 1):
            if neighbor_level_idx < level_idx:
                transfer_indices = lambda idx: idx // (2 ** (level_idx - neighbor_level_idx))
            else:
                transfer_indices = lambda idx: idx
            neighbor_level = grid.levels[neighbor_level_idx]

            # Find the neighbor-level block that contains the requested cells
            # Because the fine-level block is properly nested within the neighbor level blocks
            # at each level, we can simply find the neighbor block that contains the first ghost cell
            neighbor_origin_of_fine_block = jax.tree.map(transfer_indices, fine_ghost_cell_first_index)
            neighbor_block_idx = jax.tree.map(
                    lambda s, b_s: s // b_s,
                    neighbor_origin_of_fine_block, grid.level_specs[neighbor_level_idx].block_shape)
            neighbor_block_origin = jax.tree.map(
                    lambda s, b_s: s * b_s,
                    neighbor_block_idx, grid.level_specs[neighbor_level_idx].block_shape)

            # Calculate the position in the neighbor block of the requested cells
            neighbor_ghost_cell_indices = jax.tree.map(transfer_indices, fine_ghost_cell_indices)
            neighbor_block_ghost_cell_offsets = jax.tree.map(
                    lambda a, b: a - b, neighbor_ghost_cell_indices, neighbor_block_origin)

            neighbor_block_active_idx = neighbor_level.block_index_map[neighbor_block_idx]
            neighbor_values = self.level_values[neighbor_level_idx][neighbor_block_active_idx, ...]
            neighbor_ghost_cells = neighbor_values[neighbor_block_ghost_cell_offsets]
            result = jax.lax.cond(neighbor_block_active_idx == -1,
                                  lambda: result,
                                  lambda: jnp.reshape(neighbor_ghost_cells, result.shape))

        return result


    def ghost_cells_boundary(self, level_idx, block_origin, dim, direction, boundary_condition):
        grid = self.grid
        level = grid.levels[level_idx]
        level_spec = grid.level_specs[level_idx]
        center_coords = grid.level_coordinates_center[level_idx]
        ghost_cells_shape = [*level_spec.block_shape]
        ghost_cells_shape[dim] = 1
        result = jnp.zeros(ghost_cells_shape)

        one_to_ndim = tuple(range(grid.n_dims))

        block_indices = jax.tree.map(lambda s, b: s // b,
                                     block_origin, level_spec.block_shape)
        block_active_idx = level.block_index_map[block_indices]

        fine_copyout_cell_relative_indices = level_spec.block_copyout_cell_relative_indices(dim, direction)
        fine_copyout_cell_indices = jax.tree.map(
                lambda a, b: a + b, block_origin, fine_copyout_cell_relative_indices)
        copyout_cell_values = self.level_values[level_idx][block_active_idx, *fine_copyout_cell_relative_indices]
        copyout_cell_values = jnp.reshape(copyout_cell_values,
                                          jax.tree.map(lambda a: a.shape[0], fine_copyout_cell_relative_indices))

        coords = jax.tree.map(
                lambda idx, coords, axis: jnp.take(coords, idx, axis=axis),
                fine_copyout_cell_indices, center_coords, one_to_ndim)

        return boundary_condition(coords, copyout_cell_values)


    def copyout_cells(self, level_idx, block_origin, dim, direction):
        grid = self.grid
        level = grid.levels[level_idx]
        level_spec = grid.level_specs[level_idx]
        center_coords = grid.level_coordinates_center[level_idx]
        ghost_cells_shape = [*level_spec.block_shape]
        ghost_cells_shape[dim] = 1
        result = jnp.zeros(ghost_cells_shape)

        one_to_ndim = tuple(range(grid.n_dims))

        block_indices = jax.tree.map(lambda s, b: s // b,
                                     block_origin, level_spec.block_shape)
        block_active_idx = level.block_index_map[block_indices]

        fine_copyout_cell_relative_indices = level_spec.block_copyout_cell_relative_indices(dim, direction)
        copyout_cell_values = self.level_values[level_idx][block_active_idx][fine_copyout_cell_relative_indices]
        copyout_cell_values = jnp.reshape(copyout_cell_values,
                                          jax.tree.map(lambda a: a.shape[0], fine_copyout_cell_relative_indices))

        return copyout_cell_values



def normal_vector(dim, direction, n_dims):
    result = [0.0]*n_dims
    result[dim] = -1.0 if direction == 'left' else 1.0
    return jnp.array(result)

def exchange_crosslevel_fluxes(crosslevel_fluxes, level, dim, level_spec, n_dims):
    if n_dims == 1:
        return exchange_crosslevel_fluxes_1d(crosslevel_fluxes, level, level_spec, dim)
    elif n_dims == 2:
        return exchange_crosslevel_fluxes_2d(crosslevel_fluxes, level, level_spec, dim)


def exchange_crosslevel_fluxes_1d(crosslevel_fluxes, level, level_spec, dim):
    assert dim == 0
    block_array_shape = level_spec.block_array_shape
    return do_exchange_crosslevel_fluxes(crosslevel_fluxes, level, dim,
        (jnp.arange(block_array_shape[0]-1), 1+jnp.arange(block_array_shape[0]-1)), (0, -1))


def exchange_crosslevel_fluxes_2d(crosslevel_fluxes, level, level_spec, dim):
    block_array_shape = level_spec.block_array_shape
    block_shape = level_spec.block_shape
    if dim == 0:
        return do_exchange_crosslevel_fluxes(crosslevel_fluxes, level, 0,
            ((jnp.arange(block_array_shape[0]-1), jnp.arange(block_array_shape[1])),
             (1+jnp.arange(block_array_shape[0]-1), jnp.arange(block_array_shape[1]))),
            ((0, jnp.arange(block_shape[1])), (-1, jnp.arange(block_shape[1]))))
    elif dim == 1:
        return do_exchange_crosslevel_fluxes(crosslevel_fluxes, level, 1,
            ((jnp.arange(block_array_shape[0]), jnp.arange(block_array_shape[1]-1)),
             (jnp.arange(block_array_shape[0]), 1+jnp.arange(block_array_shape[1]-1))),
            ((jnp.arange(block_shape[0]), 0), (jnp.arange(block_shape[0]), -1)))


def do_exchange_crosslevel_fluxes(crosslevel_fluxes, level, dim,
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

        print("leftright", leftright_ixs)
        print("rightleft", rightleft_ixs)

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

    crosslevel_fluxes, _ = jax.lax.scan(maybe_coalesce_pair, crosslevel_fluxes,
        (pair_left_active_indices, pair_right_active_indices))
    return crosslevel_fluxes



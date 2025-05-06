import jax.numpy as jnp
import equinox as eqx
import jax
from jaxtyping import Array, Bool, PyTree, PyTreeDef
import math
from functools import partial


class AMRLevelSpec(eqx.Module):
    n_blocks: int
    block_array_shape: PyTree
    block_shape: PyTree
    level_shape: PyTree

    def __init__(self, level, L0_shape, n_blocks, block_shape):
        level_shape = jax.tree.map(lambda s: s * (2**level), L0_shape)

        self.n_blocks = n_blocks
        self.block_array_shape = jax.tree.map(lambda s1, s2: int(s1 / s2), level_shape, block_shape)
        self.block_shape = block_shape
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
        return AMRGrid(self.n_dims, levels, self.level_specs, self.level_coordinates_center)


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

    def __init__(self, n_dims, levels, level_specs, level_coordinates_center):
        self.n_dims = n_dims
        self.levels = levels
        self.level_specs = level_specs
        self.level_coordinates_center = level_coordinates_center


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
            print(level.block_indices[0].shape)
            print(mask.shape)
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


    def ghost_cells(self, level_idx, block_origin, dim, direction, boundary_condition):
        block_indices = jax.tree.map(lambda s, shape: s // shape,
                                     block_origin, self.grid.level_specs[level_idx].block_shape)
        if direction == 'left':
            return jax.lax.cond(block_indices[dim] == 0,
                                lambda: self.ghost_cells_boundary(level_idx, block_origin, dim, direction, boundary_condition),
                                lambda: self.ghost_cells_interior(level_idx, block_origin, dim, direction))
        elif direction == 'right':
            return jax.lax.cond(block_indices[dim] == grid.level_specs[level_idx].block_array_shape[dim],
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

        for coarse_level_idx in range(level_idx):
            decimate = lambda idx: idx // (2 ** (level_idx - coarse_level_idx))
            coarse_level = grid.levels[coarse_level_idx]

            # Find the coarse-level block that contains the requested cells
            # Because the fine-level block is properly nested within the coarse level blocks
            # at each level, we can simply find the coarse block that contains the first ghost cell
            coarse_origin_of_fine_block = jax.tree.map(decimate, fine_ghost_cell_first_index)
            coarse_block_idx = jax.tree.map(
                    lambda s, b_s: s // b_s,
                    coarse_origin_of_fine_block, grid.level_specs[coarse_level_idx].block_shape)
            coarse_block_origin = jax.tree.map(
                    lambda s, b_s: s * b_s,
                    coarse_block_idx, grid.level_specs[coarse_level_idx].block_shape)

            # Calculate the position in the coarse block of the requested cells
            coarse_ghost_cell_indices = jax.tree.map(decimate, fine_ghost_cell_indices)
            coarse_block_ghost_cell_offsets = jax.tree.map(
                    lambda a, b: a - b, coarse_ghost_cell_indices, coarse_block_origin)

            coarse_block_active_idx = coarse_level.block_index_map[coarse_block_idx]
            coarse_values = self.level_values[coarse_level_idx][coarse_block_active_idx, ...]
            coarse_ghost_cells = coarse_values[coarse_block_ghost_cell_offsets]
            result = jax.lax.cond(coarse_block_active_idx == -1,
                                  lambda: result,
                                  lambda: jnp.reshape(coarse_ghost_cells, result.shape))

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
        jax.debug.print("{}", fine_copyout_cell_indices)
        copyout_cell_values = self.level_values[level_idx][block_active_idx][fine_copyout_cell_relative_indices]
        copyout_cell_values = jnp.reshape(copyout_cell_values,
                                          jax.tree.map(lambda a: a.shape[0], fine_copyout_cell_relative_indices))
        jax.debug.print("{}", copyout_cell_values)

        coords = jax.tree.map(
                lambda idx, coords, axis: jnp.take(coords, idx, axis=axis),
                fine_copyout_cell_indices, center_coords, one_to_ndim)

        return boundary_condition(coords, copyout_cell_values)
        


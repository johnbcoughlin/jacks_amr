import jax.numpy as jnp
import equinox as eqx
import jax
from jaxtyping import Array, Bool, PyTree, PyTreeDef
import math
from functools import partial


class AMRLevelSpec():
    def __init__(self, level, L0_shape, n_blocks, block_shape):
        level_shape = jax.tree.map(lambda s: s * (2**level), L0_shape)

        self.n_blocks = n_blocks
        self.block_array_shape = jax.tree.map(lambda s1, s2: int(s1 / s2), level_shape, block_shape)
        self.block_shape = block_shape
        self.level_shape = level_shape


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
        return AMRGrid(levels)


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
    levels: list[AMRLevel]

    def __init__(self, levels):
        self.levels = levels


class AMRGridFunction():
    def __init__(self, grid):
        pass



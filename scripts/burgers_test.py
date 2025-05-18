import sys
sys.path.append("src")

from jacks_amr import amr
import jax.numpy as jnp
import jax
import equinox as eqx
import matplotlib.pyplot as plt
import matplotlib as mpl
import timeit

from jacks_amr.flux_divergence import flux_divergence
from jacks_amr.refinement import refine_grid_for_function
from jacks_amr.indicators import approximate_gradient_indicator

n_levels = 5

L0_shape = (60,)

level_specs = [amr.AMRLevelSpec(0, L0_shape, 1, L0_shape)] + [
    amr.AMRLevelSpec(i, L0_shape, min(80, 9*(2**i)), (2,)) for i in range(1, 5)
]

AMR = amr.AMRGridFactory(5, 1, L0_shape,
                         (0.,), (2. * jnp.pi,), level_specs)

def f_init(x):
    return jnp.sin(x) + 0.1


def flux(q_in, q_out, normal):
    F = 0.5 * (q_in**2 + q_out**2)
    jump = (q_out - q_in)
    return F - 0.5 * jnp.maximum(jnp.abs(q_out), jnp.abs(q_in)) * jump

copyout_bcs = lambda coords, copyout_values: copyout_values

grid = AMR.base_grid()
f0 = grid.approximate(f_init)


dt = 0.01
t = 0.0
f = f0
T = 3.0
while t < T:
    div_F = flux_divergence(f, flux, copyout_bcs)
    new_values = jax.tree.map(
        lambda f, df: f - df * dt,
        f.level_values, div_F.level_values)
    f = amr.AMRGridFunction(f.grid, new_values)
    #f = refine_grid_for_function(f, copyout_bcs, approximate_gradient_indicator)
    t += dt
    
grid = f.grid
x = grid.level_coordinates_center[0][0]
plt.plot(x, f0.level_values[0][0, ...])

print([level.n_active for level in grid.levels])

all_xs = []
all_vals = []
levels = []
for i in range(n_levels-1):
    for block_active_idx in range(grid.levels[i].n_active):
        block_idx = grid.levels[i].block_indices[0][block_active_idx]
        s = grid.level_specs[i].block_shape[0]
        for cell_idx in range(block_idx*s, (block_idx+1)*s):
            fine_cell = cell_idx*2
            fine_s = grid.level_specs[i+1].block_shape[0]
            fine_block_index = (fine_cell // fine_s)
            fine_is_active = grid.levels[i+1].block_index_map[fine_block_index] != -1
            if not fine_is_active:
                levels.append(i)
                all_xs.append(grid.level_coordinates_center[i][0][cell_idx])
                all_vals.append(f.level_values[i][block_active_idx, cell_idx])
                
                
print(levels)
all_xs = jnp.array(all_xs)
all_vals = jnp.array(all_vals)
levels = jnp.array(levels)

plt.scatter(all_xs, all_vals, c=levels)

plt.show()
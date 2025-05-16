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

L0_shape = (6, 6)

level_specs = [amr.AMRLevelSpec(0, L0_shape, 1, (6, 6))] + [
    amr.AMRLevelSpec(i, L0_shape, min(20, 9*(2**i)), (2, 2)) for i in range(1, 5)
]

AMR = amr.AMRGridFactory(5, 1, (6, 6),
                         (0., 0.), (1., 1.), level_specs)

def f_init(x, y):
    return jnp.tanh(-(jnp.sqrt(x**2 + y**2) - 0.7) / 0.1)

def criterion(coarse, fine):
    return jnp.sqrt(jnp.sum(jnp.abs(coarse - fine)**2))
    
grid = AMR.base_grid()
f0 = grid.approximate(f_init)

def flux(q_in, q_out, normal):
    u = jnp.array([0.4, 0.9])
    return 0.5 * (q_in + q_out) + (jnp.dot(u, normal))
    
copyout_bcs = lambda coords, copyout_values: copyout_values

f = refine_grid_for_function(f0, copyout_bcs, approximate_gradient_indicator)

print(f.grid.levels[1].block_indices)

sys.exit(0)
dt = 0.001
t = 0.0
f = f0
while t < 40*dt:
    print(t)
    for level_idx in range(n_levels):
        print("Level ", level_idx, ": ", f.grid.levels[level_idx].n_active)
    div_F = flux_divergence(f, flux, copyout_bcs)
    new_values = jax.tree.map(
        lambda f, df: f - df * dt,
        f.level_values, div_F.level_values)
    f = amr.AMRGridFunction(f.grid, new_values)
    t += dt
    
    f = refine_grid_for_function(f, copyout_bcs, approximate_gradient_indicator)
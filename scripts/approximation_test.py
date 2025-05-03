import sys
sys.path.append("src")

from jacks_amr import amr
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import matplotlib as mpl
import timeit

n_levels = 5

L0_shape = (6, 6)

level_specs = [amr.AMRLevelSpec(0, L0_shape, 1, (6, 6))] + [
        amr.AMRLevelSpec(i, L0_shape, min(200, 9*(4**i)), (2, 2)) for i in range(1, 5)
]

AMR = amr.AMRGridFactory(5, 1, (6, 6),
                         (0., 0.), (1., 1.), level_specs)


def f(x, y):
    return jnp.tanh(-(jnp.sqrt(x**2 + y**2) - 0.7) / 0.1)


def criterion(coarse, fine):
    return jnp.sqrt(jnp.sum(jnp.abs(coarse - fine)**2))


grid = AMR.refine_to_approximate(f, criterion)

elapsed = timeit.timeit(lambda: AMR.refine_to_approximate(f, criterion), number=10)
print("elapsed:", elapsed)

finest = f(*AMR.level_coordinates_center[-1])

print(f(*AMR.level_coordinates_center[0]))
print(grid.levels[1].block_index_map)
print(grid.levels[1].block_indices)

fig, ax = plt.subplots()

ax.imshow(finest, extent=(0., 1., 0., 1.),
          origin='lower')

for i, level in enumerate(grid.levels):
    dx = 1.0 / 6 / (2**i)
    spec = AMR.level_specs[i]
    x, y = level.block_indices
    coords = AMR.level_coordinates_lower[i]
    for j in range(level.n_active):
        block_indices = jax.tree.map(lambda a: a[j], level.block_indices)
        for k1 in range(spec.block_shape[0]):
            for k2 in range(spec.block_shape[1]):
                s1 = block_indices[0] + k1
                s2 = block_indices[1] + k2
                rect = mpl.patches.Rectangle(
                        (coords[0].flatten()[s1], coords[1].flatten()[s2]), dx, dx,
                        linewidth=1,
                        edgecolor='r',
                        facecolor='none')
                ax.add_patch(rect)



plt.show()

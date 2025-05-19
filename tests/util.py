import sys
sys.path.append("src")

from jacks_amr import amr
import jax.numpy as jnp

def construct_example_grid():
    n_levels = 4
    L0_shape = (2, 2)
    level_specs = [
        amr.AMRLevelSpec(0, L0_shape, 1, (2, 2)),
        amr.AMRLevelSpec(1, L0_shape, 4, (2, 2)),
        amr.AMRLevelSpec(2, L0_shape, 4, (2, 2)),
        amr.AMRLevelSpec(3, L0_shape, 4, (4, 4)),
    ]
    AMR = amr.AMRGridFactory(4, 1, L0_shape, (0., 0.), (1., 1.),
                             level_specs)

    levels = [
        amr.AMRLevel(jnp.array([[0]]).T,
                     1,
                     (jnp.array([0]),
                      jnp.array([0])),
                     jnp.ones(1, bool)
        ),
        amr.AMRLevel(jnp.array([[-1, -1],
                                [0, -1]]).T,
                     1,
                     (jnp.array([0, -1, -1, -1]),
                      jnp.array([1, -1, -1, -1])),
                     jnp.zeros(4, bool).at[0].set(True)
        ),
        amr.AMRLevel(jnp.array([[-1, -1, -1, -1],
                                [-1, -1, -1, -1],
                                [ 0,  1, -1, -1],
                                [-1, -1, -1, -1]]).T,
                     2,
                     (jnp.array([0, 1, -1, -1]),
                      jnp.array([2, 2, -1, -1])),
                     jnp.zeros(4, bool).at[0:2].set(True)
        ),
        amr.AMRLevel(jnp.array([[-1, -1, -1, -1],
                                [-1, -1, -1, -1],
                                [-1,  0, -1, -1],
                                [-1, -1, -1, -1]]).T,
                     1,
                     (jnp.array([1, -1, -1, -1]),
                      jnp.array([2, -1, -1, -1])),
                      jnp.zeros(4, bool).at[0].set(True)
                     )
    ]

    grid = amr.AMRGrid(2, levels, level_specs, AMR.level_coordinates_center, AMR)
    return grid

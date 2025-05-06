import sys
sys.path.append("src")

from jacks_amr import amr
import jax.numpy as jnp

def construct_example_grid():
    n_levels = 4
    L0_shape = (2, 2)
    level_specs = [
        amr.AMRLevelSpec(0, L0_shape, 1, (2, 2)),
        amr.AMRLevelSpec(1, L0_shape, 3, (2, 2)),
        amr.AMRLevelSpec(2, L0_shape, 3, (2, 2)),
        amr.AMRLevelSpec(3, L0_shape, 3, (4, 4)),
    ]
    AMR = amr.AMRGridFactory(4, 1, L0_shape, (0., 0.), (1., 1.),
                             level_specs)

    levels = [
        amr.AMRLevel(jnp.array([[0]]).T,
                     1,
                     (jnp.array([0]), 
                      jnp.array([0]))),
        amr.AMRLevel(jnp.array([[-1, -1],
                                [0, -1]]).T,
                     1,
                     (jnp.array([0, -1, -1, -1]),
                      jnp.array([1, -1, -1, -1]))),
        amr.AMRLevel(jnp.array([[-1, -1, -1, -1],
                                [-1, -1, -1, -1],
                                [ 0,  1, -1, -1],
                                [-1, -1, -1, -1]]).T,
                     2,
                     (jnp.array([0, 1, -1, -1]),
                      jnp.array([2, 2, -1, -1]))),
        amr.AMRLevel(jnp.array([[-1, -1, -1, -1],
                                [-1, -1, -1, -1],
                                [-1,  0, -1, -1],
                                [-1, -1, -1, -1]]).T,
                     1,
                     (jnp.array([1, -1, -1, -1]),
                      jnp.array([2, -1, -1, -1])))
    ]

    grid = amr.AMRGrid(2, levels, level_specs, AMR.level_coordinates_center)
    return grid


def test_example_constructor():
    grid = construct_example_grid()
    
    f = lambda x, y: jnp.tanh(-(jnp.sqrt(x**2 + y**2) - 0.7) / 0.1)
    q = grid.approximate(lambda x, y: jnp.tanh(-(jnp.sqrt(x**2 + y**2) - 0.7) / 0.1))
    
    ghost_cells = q.ghost_cells(3, (4, 8), 0, 'left')
    assert jnp.allclose(ghost_cells, f(jnp.array([3/16, 3/16, 3/16, 3/16]),
                                       jnp.array([9/16, 9/16, 11/16, 11/16])))

import sys
sys.path.append("src")
sys.path.append("tests")

from jacks_amr import amr
from util import construct_example_grid
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)


def test_interior_ghost_cells():
    grid = construct_example_grid()
    
    f = lambda x, y: jnp.tanh(-(jnp.sqrt(x**2 + y**2) - 0.7) / 0.1)
    q = grid.approximate(f)
    
    ghost_cells = q.ghost_cells_interior(3, (4, 8), 0, 'left')
    assert jnp.allclose(ghost_cells, f(jnp.array([3/16, 3/16, 3/16, 3/16]),
                                       jnp.array([9/16, 9/16, 11/16, 11/16])))


def test_exterior_ghost_cells_dont_throw():
    grid = construct_example_grid()
    f = lambda x, y: jnp.tanh(-(jnp.sqrt(x**2 + y**2) - 0.7) / 0.1)
    q = grid.approximate(f)
    
    ghost_cells = q.ghost_cells_interior(2, (0, 4), 0, 'left')
    ghost_cells = q.ghost_cells_interior(1, (0, 2), 1, 'right')


def test_ghost_cells_boundary():
    grid = construct_example_grid()
    f = lambda x, y: jnp.tanh(-(jnp.sqrt(x**2 + y**2) - 0.7) / 0.1)
    q = grid.approximate(f)

    bc = lambda coords, copyout_values: -copyout_values
    ghost_cells = q.ghost_cells_boundary(2, (0, 4), 0, 'left', bc)
    print(ghost_cells)
    print(f(jnp.array([1/16, 1/16]),
                                        jnp.array([9/16, 11/16])))
    assert jnp.allclose(ghost_cells, -f(jnp.array([1/16, 1/16]),
                                        jnp.array([9/16, 11/16])))


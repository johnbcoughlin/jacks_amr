import sys
sys.path.append("src")
sys.path.append("tests")

from jacks_amr import amr
import jax.numpy as jnp

def test_1d_base_grid():
    n_levels = 5
    
    L0_shape = (60,)
    
    level_specs = [amr.AMRLevelSpec(0, L0_shape, 1, (60,))] + [
        amr.AMRLevelSpec(i, L0_shape, min(80, 9*(2**i)), (2,)) for i in range(1, 5)
    ]
    
    AMR = amr.AMRGridFactory(5, 1, L0_shape,
                             (0.,), (2. * jnp.pi,), level_specs)
    
    assert AMR.level_coordinates_center[0][0].shape == (60,)
    
    grid = AMR.base_grid()
    
    assert grid.levels[0].block_index_map.shape == (1,)
    assert grid.levels[0].block_indices[0].shape == (1,)
    
    f = grid.approximate(lambda x: jnp.cos(x))
    
    assert f.level_values[0].shape == (1, 60)
    
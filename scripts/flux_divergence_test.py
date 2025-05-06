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
        amr.AMRLevelSpec(i, L0_shape, min(80, 9*(4**i)), (2, 2)) for i in range(1, 5)
]

AMR = amr.AMRGridFactory(5, 1, (6, 6),
                         (0., 0.), (1., 1.), level_specs)


def f(x, y):
    return jnp.tanh(-(jnp.sqrt(x**2 + y**2) - 0.7) / 0.1)


def criterion(coarse, fine):
    return jnp.sqrt(jnp.sum(jnp.abs(coarse - fine)**2))


grid = AMR.refine_to_approximate(f, criterion)
approximated = grid.approximate(f)

def flux(q_in, q_out, normal):
    u = jnp.array([0.4, 0.9])
    return 0.5 * (q_in + q_out) + (jnp.dot(u, normal))

copyout_bcs = lambda coords, copyout_values: copyout_values

amr.flux_divergence(approximated, flux, copyout_bcs)

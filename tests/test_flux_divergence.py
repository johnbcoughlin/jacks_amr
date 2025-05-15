import sys
sys.path.append("src")
sys.path.append("tests")

from jacks_amr import amr
from jacks_amr.flux_divergence import flux_divergence

from util import construct_example_grid
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

def test_flux_divergence():
    grid = construct_example_grid()

    f = lambda x, y: jnp.tanh(-(jnp.sqrt(x**2 + y**2) - 0.7) / 0.1)
    q = grid.approximate(f)

    def flux(q_in, q_out, normal):
        u = jnp.array([0.4, 0.9])
        return 0.5 * (q_in + q_out) * (jnp.dot(u, normal))

    copyout_bcs = lambda coords, copyout_values: copyout_values

    div_F = flux_divergence(q, flux, copyout_bcs)

    a1 = f(0.25, 0.25)
    c1 = f(1/16, 9/16)
    c2 = f(3/16, 9/16)
    c3 = f(1/16, 11/16)
    c4 = f(3/16, 11/16)

    up = jnp.array([0., -1.])
    right = jnp.array([1., 0.])

    div_F_c1_expected = flux(c1, a1, up) / 8 + flux(c1, c3, -up) / 8 \
        + flux(c1, c2, right) / 8 + flux(c1, c1, -right) / 8

    assert jnp.isclose(div_F_c1_expected, div_F[2][0, 0, 0])

    d1 = f(9/32, 17/32)
    d2 = f(9/32, 19/32)
    div_F_c2_expected = flux(c2, a1, up) / 8 + flux(c2, c1, -right) / 8 \
        + flux(c2, c4, -up) / 8 + flux(c2, d1, right) / 16 + flux(c2, d2, right) / 16
    assert jnp.isclose(div_F_c2_expected, div_F[2][0, 1, 0])

    b1 = f(1/8, 7/8)
    b2 = f(3/8, 7/8)

    div_F_b1_expected = flux(b1, c3, up) / 8 + flux(b1, c4, up) / 8 \
        + flux(b1, b2, right) / 4 + flux(b1, b1, -right) / 4 + flux(b1, b1, -up) / 4

    assert jnp.isclose(div_F_b1_expected, div_F[1][0, 0, 1])

    a2 = f(3/4, 1/4)
    a4 = f(3/4, 3/4)
    d1234 = f(jnp.array([15/32, 15/32, 15/32, 15/32]), jnp.array([17/32, 19/32, 21/32, 23/32]))
    div_F_a4_expected = flux(a4, a2, up) / 2 + flux(a4, a4, right) / 2 + flux(a4, a4, -up) / 2 \
        + flux(a4, b2, -right) / 4 + jnp.sum(flux(a4, d1234, -right)) / 16

    assert jnp.isclose(div_F_a4_expected, div_F[0][0, 1, 1])

import sys
sys.path.append("src")
sys.path.append("tests")

from jacks_amr import amr
from jacks_amr.flux_divergence import flux_divergence

from util import construct_example_grid
import jax.numpy as jnp
import jax
import equinox as eqx

jax.config.update("jax_enable_x64", True)

def test_flux_divergence_1d():
    n_levels = 5
    L0_shape = (20,)
    level_specs = [amr.AMRLevelSpec(0, L0_shape, 1, L0_shape)] + [
        amr.AMRLevelSpec(i, L0_shape, min(1, 9*(2**i)), (2,)) for i in range(1, 5)
    ]
    AMR = amr.AMRGridFactory(5, 1, L0_shape,
                            (0.,), (2. * jnp.pi,), level_specs)
    
    dx = 2*jnp.pi / 20
    xs = jnp.linspace(dx/2, 2*jnp.pi-dx/2, 20)
    def f_init(x):
        return jnp.sin(x) + 0.1
    
    def flux(q_in, q_out, normal):
        F = 0.5 * (q_in**2 + q_out**2)
        jump = (q_out - q_in)
        return F - 0.5 * jnp.maximum(jnp.abs(q_out), jnp.abs(q_in)) * jump
    
    copyout_bcs = lambda coords, copyout_values: copyout_values
    
    grid = AMR.base_grid()
    f0 = grid.approximate(f_init)
    div_F = flux_divergence(f0, flux, copyout_bcs)
    
    f9 = f_init(xs[9])
    f10 = f_init(xs[10])
    f11 = f_init(xs[11])
    right = jnp.array([1.])
    assert div_F.level_values[0][0, 10] == (flux(f10, f11, right) - flux(f9, f10, right)) / dx
    
    grid, _ = grid.with_block_active(1, (10,))
    
    f = eqx.tree_at(
        where=lambda t: t.grid,
        pytree=f0.with_block_refined(1, 0, (10,)),
        replace_fn=lambda t: grid)
    assert f.level_values[1][0, 0] == f10
    assert f.level_values[1][0, 1] == f10
    
    ghost_cell = f.ghost_cells_interior(1, (20,), 0, 'left')
    
    div_F = flux_divergence(f, flux, copyout_bcs)
    expected = (flux(f10, f10, right) - flux(f9, f10, right)) / (dx / 2)
    assert div_F.level_values[1][0, 0] == (flux(f10, f10, right) - flux(f9, f10, right)) / (dx / 2)


def test_flux_divergence_2d():
    grid = construct_example_grid()

    f = lambda x, y: jnp.tanh(-(jnp.sqrt(x**2 + y**2) - 0.7) / 0.1)
    q = grid.approximate(f)
    
    # Cell areas
    A0 = 1/4
    A1 = A0/4
    A2 = A1/4
    A3 = A2/4

    def flux(q_in, q_out, normal):
        u = jnp.array([0.4, 0.9])
        return 0.5 * (q_in + q_out) * (jnp.dot(u, normal))

    copyout_bcs = lambda coords, copyout_values: copyout_values

    div_F = flux_divergence(q, flux, copyout_bcs).level_values

    a1 = f(0.25, 0.25)
    c1 = f(1/16, 9/16)
    c2 = f(3/16, 9/16)
    c3 = f(1/16, 11/16)
    c4 = f(3/16, 11/16)

    up = jnp.array([0., -1.])
    right = jnp.array([1., 0.])

    div_F_c1_expected = (flux(c1, a1, up) / 8 + flux(c1, c3, -up) / 8 \
        + flux(c1, c2, right) / 8 + flux(c1, c1, -right) / 8) / A2

    assert jnp.isclose(div_F_c1_expected, div_F[2][0, 0, 0])

    d1 = f(9/32, 17/32)
    d2 = f(9/32, 19/32)
    div_F_c2_expected = (flux(c2, a1, up) / 8 + flux(c2, c1, -right) / 8 \
        + flux(c2, c4, -up) / 8 + flux(c2, d1, right) / 16 + flux(c2, d2, right) / 16) / A2
    assert jnp.isclose(div_F_c2_expected, div_F[2][0, 1, 0])

    b1 = f(1/8, 7/8)
    b2 = f(3/8, 7/8)

    div_F_b1_expected = (flux(b1, c3, up) / 8 + flux(b1, c4, up) / 8 \
        + flux(b1, b2, right) / 4 + flux(b1, b1, -right) / 4 + flux(b1, b1, -up) / 4) / A1

    assert jnp.isclose(div_F_b1_expected, div_F[1][0, 0, 1])

    a2 = f(3/4, 1/4)
    a4 = f(3/4, 3/4)
    d1234 = f(jnp.array([15/32, 15/32, 15/32, 15/32]), jnp.array([17/32, 19/32, 21/32, 23/32]))
    div_F_a4_expected = (flux(a4, a2, up) / 2 + flux(a4, a4, right) / 2 + flux(a4, a4, -up) / 2 \
        + flux(a4, b2, -right) / 4 + jnp.sum(flux(a4, d1234, -right)) / 16) / A0

    assert jnp.isclose(div_F_a4_expected, div_F[0][0, 1, 1])

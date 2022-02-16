import pytest
import jax.numpy as jnp
from jax import random
from rnpe.bijections import Affine, Permute

def test_Affine():
    d = 5
    b = Affine()
    x = jnp.arange(5)
    scale = jnp.full((d,), 2)
    log_scale = jnp.log(scale)
    loc = jnp.ones((5,))

    y = b.transform(x, loc, log_scale)
    x_reconstructed = b.inverse(y, loc, log_scale)
    assert pytest.approx(x, x_reconstructed)
    assert (x != y).all()

def test_Permute():
    x = jnp.arange(4)
    permute = Permute(jnp.array([3,2,1,0]))
    y = permute.transform(x)
    x_reconstructed = permute.inverse(y)
    assert pytest.approx(x, x_reconstructed)
    assert (x != y).sum()


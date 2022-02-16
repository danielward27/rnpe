from rnpe.masks import coupling_mask
import jax.numpy as jnp
import pytest

def test_coupling_mask():
    expected = jnp.concatenate((jnp.ones((10,2)), jnp.zeros((10, 3))), axis=1)
    result = jnp.ones((10, 5))*coupling_mask(2, 5)
    assert (expected == result).all()

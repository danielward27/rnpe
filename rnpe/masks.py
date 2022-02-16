from jax import jit
import jax.numpy as jnp

def coupling_mask(d: int, D: int):
    return jnp.concatenate((jnp.ones(d), jnp.zeros(D-d)))
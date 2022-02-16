

# %%
import jax
from jax import random
import jax.numpy as jnp
import equinox as eqx
import flax.linen as nn


def coupling_mask(d: int, D: int):
    return jnp.concatenate((jnp.ones(d), jnp.zeros(D-d)))


# %%
jnp.ones((5, 10))*coupling_mask(2, 10)


# %%

# Batched coupling mask



def block_diag_mask(d: int, b: int):
    """Masking matrix, for block diagonal elements.

    Args:
        d (int): Dimension of matrix.
        b (int): Dimension of blocks.
    """
    ones = jnp.ones((b, b))
    ones_list = [ones for _ in range(d//b)]
    mask = jax.scipy.linalg.block_diag(*ones_list)
    return mask


def tril_block_mask(d: int, b: int):
    """Masking matrix selecting lower triangular elements, excluding block diagonal

    Args:
        d (int): Dimension of matrix.
        b (int): Dimension of blocks.
    """
    mask = jnp.ones((d,d))
    bd_mask = block_diag_mask(d, b)
    mask = mask - bd_mask
    return jnp.tril(mask)


# %%
class SimpleDense(nn.Module):
  features: int
  kernel_init: Callable = nn.initializers.lecun_normal()
  bias_init: Callable = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs):
    kernel = self.param('kernel',
                        self.kernel_init, # Initialization function
                        (inputs.shape[-1], self.features))  # shape info.
    y = lax.dot_general(inputs, kernel,
                        (((inputs.ndim - 1,), (0,)), ((), ())),) # TODO Why not jnp.dot?
    bias = self.param('bias', self.bias_init, (self.features,))
    y = y + bias
    return y

# %%
# class BNAFLayer(eqx.Module):
#     "Weight matrix is "
#     weight: jnp.ndarray
#     bias: jnp.ndarray

#     def __init__(self, d, a, b, out_size, key):
#         weight_size = (a*d, b*d)
#         wkey, bkey = jax.random.split(key)
#         weight = jax.random.normal(wkey, (out_size, in_size))
#         bias = jax.random.normal(bkey, (out_size,))
#         weight = 
        
#     @jax.jit
#     def __call__(self, x):
#         W = self.W*tril_block_mask() + jnp.exp(self.W*block_diag_mask())

#         W = jnp.tril(self.W)
#         W = exp_diag(W)
#         z = self.weight @ x + self.bias

#         return z, log_abs_det







# # %%


# # %% Sample base distribution
# key = random.PRNGKey(0)
# key, subkey = random.split(key)
# z = random.normal(key, (100, 2))




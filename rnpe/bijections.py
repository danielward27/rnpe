from itertools import permutations
import jax.numpy as jnp
from jax import jit
from functools import partial
from flax import struct

class Affine:
    def transform(self, x, loc, log_scale):
        return x*jnp.exp(log_scale) + loc
        
    def transform_and_log_abs_det_jacobian(self, x, loc, log_scale):
        return self.transform(x, loc, log_scale), jnp.sum(log_scale)

    def inverse(self, y, loc, log_scale):
        return (y-loc)/jnp.exp(log_scale)

    def param_getter(self, param_flat):
        return param_flat.split(2)  # split to loc and scale


class Permute:
    permutation: jnp.ndarray  # with indices 0-d
    inverse_permutation: jnp.ndarray

    def __init__(self, permutation):
        self.permutation = permutation 
        self.inverse_permutation = jnp.argsort(permutation)  

    def __call__(self, x):
         return self.transform_and_log_abs_det_jacobian(x)

    def transform(self, x):
        return x[self.permutation]

    def transform_and_log_abs_det_jacobian(self, x):
        return x[self.permutation], jnp.array([0])

    def inverse(self, y):
        return y[self.inverse_permutation]

# %%
from tkinter import X
from typing import Callable, Sequence
from jax import jit
import jax.numpy as jnp
import equinox as eqx
import flax.linen as nn
from jax import random
from rnpe.bijections import Affine, Permute

class Conditioner(nn.Module):
    """param_getter should reshape/restructure a vector to args of a transformation"""
    features: Sequence[int]  # Last one should be number of required parameters
    param_getter: Callable = jnp.identity



    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]
    
    def __call__(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = nn.relu(x)

        return self.param_getter(x)


class CouplingLayer(nn.Module):
    """Mask is binary vector denoting untransformed variables with ones."""
    d: int  # Where to partition
    D: int # Total dimension
    hidden_features: Sequence[int]
    bijection = Affine()
    
    def setup(self):
        self.conditioner = Conditioner(
            [*self.hidden_features, (self.D-self.d)*2],
            self.bijection.param_getter
            )

    def __call__(self, x: jnp.ndarray):
        return self.transform_and_log_abs_det_jacobian(x)    
        
    def transform_and_log_abs_det_jacobian(self, x):
        x_cond, x_trans = x[:self.d], x[self.d:]
        t_args = self.conditioner(x_cond)
        y_trans, log_abs_det = self.bijection.transform_and_log_abs_det_jacobian(
            x_trans, *t_args)
        y = jnp.concatenate([x_cond, y_trans])
        return y, log_abs_det

    def transform(self, x):
        x_cond, x_trans = x[:self.d], x[self.d:]
        t_args = self.conditioner(x_cond)
        y_trans = self.bijection.transform(x_trans, *t_args)
        y = jnp.concatenate([x_cond, y_trans])
        return y

    def inverse(self, y: jnp.ndarray):
        x_cond, y_trans = y[:self.d], y[self.d:]
        t_args = self.conditioner(x_cond)
        x_trans = self.bijection.inverse(y_trans, *t_args)
        x = jnp.concatenate([x_cond, x_trans])
        return x


# class ComposeLayers(nn.Module):



class RealNVP(nn.Module):
    contitioner_features: Sequence[int]
    num_layers: int
    D: int

    def setup(self):
        d = round(self.D//2)
        layers = []
        for i in range(self.num_layers):
            permutation = random.shuffle(random.PRNGKey(i), jnp.arange(self.D))  # TODO Probably better way to handle seed/initialization here!
            layers.extend(
                [CouplingLayer(d, self.D, hidden_features=self.contitioner_features),
                Permute(permutation)]
            )
        self.layers = layers
    
    def __call__(self, x):
        log_abs_det_jac = 0
        z = x
        for layer in self.layers:
            z, log_abs_det_jac_i = layer(x)
            log_abs_det_jac += log_abs_det_jac_i
        return z, log_abs_det_jac

    def inverse(self, z):
        x = z
        for layer in reversed(self.layers):
            x, log_abs_det_jac_i = layer(x)  # TODO don't need jacobian
        return x


        



            
        
        
    
# %%
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.relu(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x

class WrappedMLP(nn.Module):
    def __init__(self, mlp : nn.Module):
        self.mlp = mlp

    def __call__(self, input):
        self.mlp.apply()

model = MLP([12, 8, 4])
batch = jnp.ones((32, 10))
variables = model.init(jax.random.PRNGKey(0), batch)
output = model.apply(variables, batch)

# %%
# Now try to train to see if works for simple transformation?





# %%
# Transformations defined for single example then vmaped if required





# %%


# class AffineCoupling(eqx.Module):
#     "Mask is binary vector denoting untransformed variables with ones."

#     @jit
#     def forward(
#         x: jnp.ndarray,
#         loc: jnp.ndarray,
#         log_scale: jnp.ndarray,
#         mask: jnp.ndarray = jnp.array([0.])
#         ):
#         return x*mask + (1-mask) * (x * jnp.exp(log_scale) + loc)   # eq 9

#     @jit
#     def inv(
#         y: jnp.ndarray,
#         loc: jnp.ndarray,
#         log_scale: jnp.ndarray,
#         mask: jnp.ndarray = jnp.array([0.])
#         ):
#         return y*mask + (1-mask) * (y-loc)/log_scale

#     @jit
#     def forward_log_abs_det_jacobian(
#         x: jnp.ndarray,
#         loc: jnp.ndarray,
#         log_scale: jnp.ndarray,
#         mask: jnp.ndarray = jnp.array([0.])):
#         return jnp.sum((1-mask)*log_scale)
        
#     @jit
#     def inv_log_abs_det_jacobian(
#         x: jnp.ndarray,
#         loc: jnp.ndarray,
#         log_scale: jnp.ndarray,
#         mask: jnp.ndarray = jnp.array([0.])
#     ):
#         return -jnp.sum((1-mask)*log_scale)





# # Maybe use flax, avoiding specifying the parameters is nice...

# affine_layer = AffineCoupling()


# # %%
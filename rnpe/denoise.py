import numpyro
import numpyro.distributions as dist
from jaxflows.flows import Flow

from numpyro.distributions import Distribution, constraints
import jax.numpy as jnp
import numpyro


class FlowDist(Distribution):
    support = constraints.real

    def __init__(self, flow: Flow):
        self.flow = flow
        super().__init__(batch_shape=(), event_shape=(flow.target_dim,))

    def sample():
        return NotImplementedError

    def log_prob(self, value):
        return self.flow.log_prob(value)


def cauchy_denoiser(y_obs, flow, scale=0.025):
    x = numpyro.sample("x", FlowDist(flow))  # Transform
    numpyro.sample("y", dist.Cauchy(x, scale), obs=y_obs)


# class NormDist(Distribution):
#     support = constraints.real

#     def __init__(self, loc: jnp.ndarray, norm: float = 0.2, scale: float = 1.5):
#         self.loc = loc
#         self.norm = norm
#         self.scale = scale
#         super().__init__(batch_shape=jnp.shape(loc), event_shape=())

#     def sample(self, key, sample_shape=()):
#         raise NotImplementedError

#     def log_prob(self, value):
#         dif = jnp.abs(value - self.loc)
#         return -(((dif ** self.norm / self.scale).sum()) ** (1 / self.norm))


# def norm_denoiser(y_obs, gaussian_noise_transfrom, *args):
#     dim = jnp.shape(y_obs)[0]
#     z = numpyro.sample("z", dist.Normal(jnp.zeros(dim), 1))  # Base dist
#     x = numpyro.deterministic("x", gaussian_noise_transfrom(z))  # Transform
#     numpyro.sample("y", NormDist(x, *args), obs=y_obs)


# def horseshoe_denoiser_model_st(y_obs, gaussian_noise_transfrom, nu_lam=5, nu_tau=5):
#     "params and inverse transform for jax flow."
#     dim = jnp.shape(y_obs)[0]

#     lambdas = numpyro.sample("lambdas", dist.StudentT(nu_lam, 0, jnp.ones(dim)))
#     tau = numpyro.sample("tau", dist.StudentT(nu_tau, 0, 1))

#     scales = lambdas * tau
#     z = numpyro.sample("z", dist.Normal(jnp.zeros(dim), 1))  # Base dist
#     x = numpyro.deterministic("x", gaussian_noise_transfrom(z))  # Transform
#     numpyro.sample("y", dist.Normal(x, scales), obs=y_obs)


#
# Tried students t instead of lambdas
# Tried fixing tau


# def spike_and_slab_denoiser_model(y_obs, gaussian_noise_transfrom, probs=None):
#     dim = jnp.shape(y_obs)[0]
#     if probs is None:
#         probs = jnp.full((dim,), 0.1)

#     mask = numpyro.sample("mask", dist.Bernoulli(probs))  # 1==slab==misspecified
#     inverse_mask = mask == 0
#     misspecified_indices = jnp.where(mask)

#     z = numpyro.sample("z", dist.Normal(jnp.zeros(dim), 1))  # Base dist
#     x = gaussian_noise_transfrom(z)
#     x = mask * x + y_obs * inverse_mask
#     x = numpyro.deterministic("x", x)

#     numpyro.sample("y", dist.Normal(x[misspecified_indices], 5), obs=y_obs)

# class MaskedNormal(dist.Distribution):
#     support = constraints.real

#     def __init__(self, normal: dist.Normal, mask: jnp.array):
#         self.normal = normal
#         self.mask = mask
#         super().__init__(batch_shape=normal.batch_shape, event_shape=())

#     def sample(self, key, sample_shape=()):
#         raise NotImplementedError

#     def log_prob(self, value):
#         return (self.normal.log_prob(value)*self.mask).sum()


# class FlowDist(dist.Distribution):
#     support = constraints.real

#     def __init__(self, flow):
#         self.flow = flow
#         super().__init__(batch_shape=(), event_shape=(flow.target_dim, ))

#     def log_prob(self, value):
#         value = jnp.expand_dims(value, axis=0)
#         return self.flow.log_prob(value)[0]

#     def sample(self, key, sample_shape=()):
#         return flow.sample(key, 1)[0]  # TODO not sure if this right?


# def spike_and_slab_denoiser_model(y_obs, gaussian_noise_transfrom, well_specified_probs):
#     dim = jnp.shape(y_obs)[0]
#     well_specified_mask = numpyro.sample(
#         "well_specified_mask", dist.Bernoulli(well_specified_probs))
#     misspecified_mask = well_specified_mask == 0
#     z = numpyro.sample("z", dist.Normal(jnp.zeros(dim), 1))  # Base dist. Assuming independent of mask?
#     x = numpyro.deterministic("x", gaussian_noise_transfrom(z)*well_specified_mask + y_obs * misspecified_mask)  # shrink well specified to y_obs
#     scales = jnp.full((dim, ), 5) # not sure about bottom two lines
#     normal = Normal(x, scales)
#     numpyro.sample("y", MaskedNormal(normal, well_specified_mask), obs=y_obs)  # TODO try switching this if not

#     # with numpyro.plate("data", len(y_obs)) as ind:
#     #     numpyro.sample("y", dist.Normal(x[ind], scales[ind]), obs=y_obs[ind])
#     # pyro plate over misspecified with normal? if uniform surely slab doesn't impact log prob. Do we need the last bit?


# def horseshoe_denoiser_model(
#     y_obs, gaussian_noise_transfrom, tau_prior_scale, min_scale
# ):
#     dim = jnp.shape(y_obs)[0]
#     lambdas = numpyro.sample("lambdas", dist.HalfCauchy(jnp.ones(dim)))
#     tau = numpyro.sample("tau", dist.HalfCauchy(tau_prior_scale))
#     scales = lambdas * tau + min_scale
#     z = numpyro.sample("z", dist.Normal(jnp.zeros(dim), 1))  # Base dist
#     x = numpyro.deterministic("x", gaussian_noise_transfrom(z))  # Transform
#     numpyro.sample("y", dist.Normal(x, scales), obs=y_obs)


# def horseshoe_denoiser_model_normal_tau(
#     y_obs, gaussian_noise_transfrom, tau_scale, min_scale
# ):
#     dim = jnp.shape(y_obs)[0]
#     lambdas = numpyro.sample("lambdas", dist.HalfCauchy(jnp.ones(dim)))
#     tau = numpyro.sample("tau", dist.Normal(0, tau_scale))
#     scales = lambdas * tau + min_scale
#     z = numpyro.sample("z", dist.Normal(jnp.zeros(dim), 1))  # Base dist
#     x = numpyro.deterministic("x", gaussian_noise_transfrom(z))  # Transform
#     numpyro.sample("y", dist.Normal(x, scales), obs=y_obs)


# def just_cauchy_denoiser(y_obs, gaussian_noise_transfrom, scale=0.05):
#     dim = jnp.shape(y_obs)[0]
#     z = numpyro.sample("z", dist.Normal(jnp.zeros(dim), 1))  # Base dist
#     x = numpyro.deterministic("x", gaussian_noise_transfrom(z))  # Transform
#     numpyro.sample("y", dist.Cauchy(x, scale), obs=y_obs)

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

def horseshoe_denoiser_model(y_obs, gaussian_noise_transfrom, tau_prior_scale, min_scale):
    "params and inverse transform for jax flow."
    dim = jnp.shape(y_obs)[0]
    lambdas = numpyro.sample("lambdas", dist.HalfCauchy(jnp.ones(dim)))
    tau = numpyro.sample("tau", dist.HalfCauchy(jnp.array(tau_prior_scale)))
    scales = lambdas * tau + min_scale
    z = numpyro.sample("z", dist.Normal(jnp.zeros(dim), 1))  # Base dist
    x = numpyro.deterministic("x", gaussian_noise_transfrom(z))  # Transform
    numpyro.sample("y", dist.Normal(x, scales), obs=y_obs)


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

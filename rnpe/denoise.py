import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

def horseshoe_denoiser_model(y_obs, gaussian_noise_transfrom, tau_prior_scale=2.5e-5):
    "params and inverse transform for jax flow."
    dim = jnp.shape(y_obs)[0]
    lambdas = numpyro.sample("lambdas", dist.HalfCauchy(jnp.ones(dim)))
    tau = numpyro.sample("tau", dist.HalfCauchy(jnp.array([tau_prior_scale])))
    scales = lambdas*tau
    z = numpyro.sample("z", dist.Normal(jnp.zeros(dim), 1))  # Base dist
    x = numpyro.deterministic(
        "x", gaussian_noise_transfrom(z))  # Transform
    numpyro.sample("y", dist.Normal(x, scales), obs=y_obs)
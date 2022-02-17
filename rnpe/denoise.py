import jax.numpy as np
import numpyro
import numpyro.distributions as dist

def horseshoe_denoiser_model(y_obs, flow, tau_prior_scale=2.5e-5):
    "params and inverse transform for jax flow."
    dim = np.shape(y_obs)[0]
    lambdas = numpyro.sample("lambdas", dist.HalfCauchy(np.ones(dim)))
    tau = numpyro.sample("tau", dist.HalfCauchy(np.array([tau_prior_scale])))
    scales = lambdas*tau
    z = numpyro.sample("z", dist.Normal(np.zeros(dim), 1))  # Base dist
    x = numpyro.deterministic("x", flow.transform_noise(inputs=np.expand_dims(z, 0))[0].squeeze())  # Transform
    numpyro.sample("y", dist.Normal(x, scales), obs=y_obs)
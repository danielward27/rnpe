# %%
from jax import random
from flowjax.flows import coupling_flow
from flowjax.bijections.transformers import RationalQuadraticSplineTransformer
from flowjax.distributions import Normal
import jax.numpy as jnp
from rnpe.metrics import robust_posterior_log_prob

from rnpe.tasks import GaussianLinear
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# %%
def test_robust_posterior_log_prob():
    num_denoised = 100
    num_posterior_samps = 100
    transformer = RationalQuadraticSplineTransformer(10, 5)
    key, subkey = random.split(random.PRNGKey(0))
    flow = coupling_flow(subkey, Normal(2), transformer, 3)
    key, subkey = random.split(key)
    denoised = random.normal(subkey, (num_denoised, 3))

    key, subkey = random.split(key)

    theta = random.normal(subkey, (num_posterior_samps, 2))
    result = robust_posterior_log_prob(flow, theta, denoised)
    assert result.shape == (theta.shape[0],)

    # test elements calculating more naively
    expected = []
    for theta_row in theta:
        probs = []
        for d_row in denoised:
            probs.append(jnp.exp(flow.log_prob(theta_row, d_row)).item())
        prob = jnp.mean(jnp.array(probs))
        expected.append(jnp.log(prob))
    expected = jnp.array(expected)
    assert jnp.all((result - expected) < 1e-5)


test_robust_posterior_log_prob()

# %%


def test_GaussianLinear_posterior():
    # Compare to MCMC samples in numpyro
    task = GaussianLinear()
    y = jnp.linspace(-1, 1, task.dim)
    mu, std = task._get_true_posterior_mu_std(y)

    def model(obs, dim=15):
        with numpyro.plate("prior_plate", dim) as i:
            theta = numpyro.sample("theta", dist.Normal(0, jnp.sqrt(0.1)))
            x = numpyro.sample("x", dist.Normal(theta, jnp.sqrt(0.1)))
            y = numpyro.sample("y", dist.Normal(x, jnp.sqrt(0.1)), obs=obs[i])

    mcmc = MCMC(NUTS(model), num_warmup=100, num_samples=20000, progress_bar=False)
    mcmc.run(random.PRNGKey(0), obs=y)
    samples = mcmc.get_samples()["theta"]

    assert (mu - samples.mean(axis=0) < 0.01).all()
    assert (std - samples.std(axis=0) < 0.01).all()

test_GaussianLinear_posterior()
# %%

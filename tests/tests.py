# %%
from jax import random
from flowjax.flows import NeuralSplineFlow
import jax.numpy as jnp
from rnpe.metrics import robust_posterior_log_prob


def test_robust_posterior_log_prob():
    num_denoised = 100
    num_posterior_samps = 100
    key, subkey = random.split(random.PRNGKey(0))
    flow = NeuralSplineFlow(subkey, 2, 3)
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

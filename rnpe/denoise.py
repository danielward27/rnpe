from flowjax.distributions import Distribution as FlowJaxDist
import jax.numpy as jnp
import numpyro
from numpyro import sample
from numpyro.distributions import (
    Distribution,
    Normal,
    Bernoulli,
    Cauchy,
    StudentT,
    Uniform,
    constraints,
)


class FlowDist(Distribution):
    "Wraps flow as numpyro compatible distribution for MCMC."
    support = constraints.real

    def __init__(self, flow: FlowJaxDist):
        self.flow = flow
        super().__init__(batch_shape=(), event_shape=(flow.dim,))

    def sample(self, key, sample_shape=()):
        raise NotImplementedError()

    def log_prob(self, value):
        batch_size = value.shape[:-1]
        value = value.reshape(-1, self.flow.dim)
        return self.flow.log_prob(value).reshape(batch_size)


def spike_and_slab_denoiser_hyperprior(
    y_obs: jnp.ndarray,
    flow: FlowJaxDist,
    spike_std: float = 0.01,
    slab_scale: float = 0.25,
):
    """Model that uses Bernoulli random variable to index between using
    a tight normal (for approximately well specified), and a Cauchy (for
    misspecified) dimensions.

    Args:
        y_obs (jnp.ndarray): Observed data.
        flow (FlowJaxDist): Flow from flowjax package.
        misspecified_prob (float, optional): Prior probability of being in slab. Defaults to 0.5.
    """
    misspecified_prob = sample("misspecified_prob", Uniform(0,1))  # Bernoulli hyperprior

    with numpyro.plate("d", len(y_obs)):
        misspecified = sample("misspecified", Bernoulli(probs=misspecified_prob))

    x = sample("x", FlowDist(flow))

    with numpyro.handlers.mask(mask=~misspecified.astype(bool)):
        sample("y_obs_w", Normal(x, spike_std), obs=y_obs)

    with numpyro.handlers.mask(mask=misspecified.astype(bool)):
        sample("y_obs_m", Cauchy(x, slab_scale), obs=y_obs)

def spike_and_slab_denoiser(
    y_obs: jnp.ndarray,
    flow: FlowJaxDist,
    spike_std: float = 0.01,
    slab_scale: float = 0.25,
    misspecified_prob: float = 0.5,
):
    """Model that uses Bernoulli random variable to index between using
    a tight normal (for approximately well specified), and a Cauchy (for
    misspecified) dimensions.

    Args:
        y_obs (jnp.ndarray): Observed data.
        flow (FlowJaxDist): Flow from flowjax package.
        misspecified_prob (float, optional): Prior probability of being in slab. Defaults to 0.5.
    """
    with numpyro.plate("d", len(y_obs)):
        misspecified = sample("misspecified", Bernoulli(probs=misspecified_prob))

    x = sample("x", FlowDist(flow))

    with numpyro.handlers.mask(mask=~misspecified.astype(bool)):
        sample("y_obs_w", Normal(x, spike_std), obs=y_obs)

    with numpyro.handlers.mask(mask=misspecified.astype(bool)):
        sample("y_obs_m", Cauchy(x, slab_scale), obs=y_obs)


def cauchy_denoiser(y_obs, flow, scale=0.025):
    x = numpyro.sample("x", FlowDist(flow))  # Transform
    numpyro.sample("y", Cauchy(x, scale), obs=y_obs)


def student_t_denoiser(y_obs, flow, scale=0.01, df=0.4):
    x = numpyro.sample("x", FlowDist(flow))
    numpyro.sample("y", StudentT(df=df, loc=x, scale=scale), obs=y_obs)



# def spike_and_slab_enumerate_denoiser(
#     obs: jnp.ndarray,
#     flow: Distribution,
#     outlier_probs: jnp.ndarray,
#     slab_std: float = 10,
# ):
#     """
#     Avoids need of specialised sampling procedures, but slow for most reasonable
#     number of dimensions. Marginalizes over Bernoulli variable.
#     """
#     d = len(obs)

#     with numpyro.plate("d", d, dim=-1):
#         error_slab = sample("error_slab", Normal(0, slab_std))

#     masked_errors = []
#     for k in range(d):
#         misspecified_k = sample(
#             f"misspecified_{k}",
#             Bernoulli(probs=outlier_probs[k]),
#             infer={"enumerate": "parallel"},
#         )
#         masked_errors.append(error_slab[None, k] * misspecified_k)
#     masked_errors = jnp.concatenate(jnp.broadcast_arrays(*masked_errors), axis=-1)

#     with numpyro.plate("obs", 1):
#         sample(
#             "y",
#             TransformedDistribution(FlowDist(flow), AffineTransform(masked_errors, 1)),
#             obs=obs,
#         )


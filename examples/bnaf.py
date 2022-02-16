
# %% Imports
import numpyro
import jax.numpy as jnp

from numpyro.distributions.transforms import (
    ComposeTransform,
    PermuteTransform,
)

from numpyro.nn.block_neural_arn import BlockNeuralAutoregressiveNN

from numpyro.distributions.flows import (
    BlockNeuralAutoregressiveTransform,
    )

from numpyro import distributions as dist
from jax import random


# %% Define flow
flows = []
layers = 2  # Flow layers
latent_dim = 2
hidden_factors = [8,8]
residual = None
prefix = "pre"

base_dist = dist.Normal(jnp.zeros(latent_dim), 1).to_event(1)
for i in range(layers):
    if i > 0:
        flows.append(PermuteTransform(jnp.arange(latent_dim)[::-1]))

    arn = BlockNeuralAutoregressiveNN(
        latent_dim, [8,8], None
        )
    
    arnn = numpyro.module(
        f"{prefix}_arn__{i}", arn, (latent_dim,)
        )
    flows.append(BlockNeuralAutoregressiveTransform(arnn))

flow = dist.TransformedDistribution(base_dist, flows)

# %%
key = random.PRNGKey(12)
arn = BlockNeuralAutoregressiveNN(
        latent_dim, hidden_factors, residual
    )
# %%
arnn = numpyro.module(
        "name", arn, (2,)
        )

# %%
arn[0](key, (2,))


# %%
# We will need to use the pyro approach, as we need to compose transforms.
output_shape, params_init = nn_initializer(key)

# %%

from numpyro.infer.autoguide import AutoBNAFNormal



AutoBNAFNormal()


from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

from numpyro.infer.autoguide import AutoBNAFNormal

# maybe get posterior isn't called until later or something... No idea.
# %%


##### %% Generate toy data #######

# from sklearn import datasets
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns

# n_samples = 1000
# X, y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
# X = StandardScaler().fit_transform(X)

# plt.title(r'Samples from $p(x_1,x_2)$')
# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')
# plt.scatter(X[:,0], X[:,1], alpha=0.5)
# plt.show()

# %%
latent_dim = 2
arn = BlockNeuralAutoregressiveNN(
        2, [8,8]
    )  # init apply tuple

arnn = numpyro.module(
    "prefix", arn, (latent_dim,)
)

# %%
from jax.experimental import stax

def encoder(hidden_dim, z_dim):
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Softplus,
        stax.FanOut(2),
        stax.parallel(
            stax.Dense(z_dim, W_init=stax.randn()),
            stax.serial(stax.Dense(z_dim, W_init=stax.randn()), stax.Exp),
        ),
    )

def model():
    encode = numpyro.module("decoder", encoder(2, 3), (4, 5))

# It just has to be in a model for this to work?
# %%
from jax.experimental import stax

def encoder(hidden_dim, z_dim):
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()),
        stax.Softplus,
        stax.FanOut(2),
        stax.parallel(
            stax.Dense(z_dim, W_init=stax.randn()),
            stax.serial(stax.Dense(z_dim, W_init=stax.randn()), stax.Exp),
        ),
    )
nn_init, nn_apply = encoder(2, 2)

# %%
from functools import partial
f = partial(nn_apply, input_shape=2)
f(key)

# %%


def nn_init_wrapper(input_shape):
    def inner(key):
        return enc_init(key, input_shape)
    return inner

nn_initializer = nn_init_wrapper(input_shape=(-1, 41))

# %%
from jax import random
key = random.PRNGKey(42)
output_shape, params_init = nn_initializer(key)

# %%
partial

# %%
numpyro.module("decoder", (nn_initializer, enc_apply), (4, 5))

# %%
# https://num.pyro.ai/en/stable/examples/vae.html


# %%
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
import jax.numpy as jnp
from jax import random

def t(z):  # My deterministic transformation
    return z+100

def model(y_obs):
    dim = jnp.shape(y_obs)[0]
    lambdas = numpyro.sample("lambdas", dist.HalfCauchy(jnp.ones(dim)))
    tau = numpyro.sample("tau", dist.HalfCauchy(jnp.array([0.001])))
    scales = lambdas*tau
    z = numpyro.sample("z", dist.Laplace(jnp.zeros(dim), 1))  # Base dist
    x = numpyro.deterministic("x", t(z))  # Transform
    numpyro.sample("y", dist.Normal(x, scales), obs=y_obs)

# %%
y_obs = jnp.array([99, 101, 108])

kernel = NUTS(model)
mcmc = MCMC(
    kernel,
    num_warmup=5000,
    num_samples=5000
)

mcmc.run(random.PRNGKey(0), y_obs)
mcmc.print_summary()

# %% Plot posterior pairplot
import seaborn as sns
import pandas as pd
import numpy as onp
samples = mcmc.get_samples()
samples = {k: onp.array(samples[k]) for k in ["lambdas", "x", "tau"]}  # Ignore z
colnames = [f"{param}{i}" for param in samples.keys() for i in range(samples[param].shape[1])]
samples_array = onp.concatenate(list(samples.values()), axis = 1)
df = pd.DataFrame(samples_array, columns = colnames)

# %%
sns.kdeplot(df["x2"])

# %%

sns.pairplot(df, vars=colnames)

# %%
# Questions: The n_eff seems ok but the sampler diverges many times (628).
# With these shrinkage type priors, does the posterior geometry cause issues with NUTS?
# Do I need to worry about these divergences and is there a better way around it?
# Finally, could I replace t with a flow? if so how would I go about that in a way that
# works with numpyro? 

# Do I need to worry about multimodality. It's possible to force multimodality with some setups.
# e.g. z sampled from a laplace, and the right distance away x

# I coded up a version using the potential_fn approach in pyro, and it seemed to work ok,
# but ran extremely slowly (~6h/1k samples). Could I expect a speed up by using numpyro,
# or is the overhead mostly unavoidable caused by the flow?
# %%

# %%
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing
import jax.numpy as np




# %% Toy data
n_samples = 10000
plot_range = [(-2, 2), (-2, 2)]
n_bins = 100

scaler = preprocessing.StandardScaler()
X, _ = datasets.make_moons(n_samples=n_samples, noise=.05)
X = scaler.fit_transform(X)
plt.hist2d(X[:, 0], X[:, 1], bins=n_bins, range=plot_range)[-1]

# %%




# %%
import flows
from jax import grad, jit, random
from jax.experimental import stax, optimizers

rng, flow_rng = random.split(random.PRNGKey(0))
input_dim = X.shape[1]
num_epochs, batch_size = 300, 100

# %%
def get_masks(input_dim, hidden_dim=64, num_hidden=1):
    masks = []
    input_degrees = np.arange(input_dim)
    degrees = [input_degrees]

    for _ in range(num_hidden + 1):
        degrees += [np.arange(hidden_dim) % (input_dim - 1)]
    degrees += [input_degrees % input_dim - 1]

    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [np.transpose(np.expand_dims(d1, -1) >= np.expand_dims(d0, 0)).astype(np.float32)]
    return masks

def masked_transform(rng, input_dim):
    masks = get_masks(input_dim, hidden_dim=64, num_hidden=1)
    act = stax.Relu
    init_fun, apply_fun = stax.serial(
        flows.MaskedDense(masks[0]),
        act,
        flows.MaskedDense(masks[1]),
        act,
        flows.MaskedDense(masks[2].tile(2)),
    )
    _, params = init_fun(rng, (input_dim,))
    return params, apply_fun

transformation = flows.Serial(*(flows.MADE(masked_transform), flows.Reverse()) * 5)

init_fun = flows.Flow(
    transformation,
    flows.Normal(),
)

params, log_pdf, sample = init_fun(flow_rng, input_dim)

from tqdm import trange
import itertools
import numpy.random as npr

# %%
def loss(params, inputs):
    return -log_pdf(params, inputs).mean()

opt_init, opt_update, get_params = optimizers.adam(step_size=1e-4)
opt_state = opt_init(params)

@jit
def step(i, opt_state, inputs):
    params = get_params(opt_state)
    gradients = grad(loss)(params, inputs)
    return opt_update(i, gradients, opt_state)

# %%
itercount = itertools.count()

for epoch in trange(num_epochs):
    permute_rng, rng = random.split(rng)
    X = random.permutation(permute_rng, X)
    for batch_index in range(0, len(X), batch_size):
        opt_state = step(next(itercount), opt_state, X[batch_index:batch_index+batch_size])
     
    params = get_params(opt_state)
    sample_rng, rng = random.split(rng)
    X_syn = sample(rng, params, X.shape[0])
    
plt.hist2d(X_syn[:, 0], X_syn[:, 1], bins=n_bins, range=plot_range)
plt.show()

params = get_params(opt_state)
# %%   ##################### Denoising  ####################
rng, flow_rng = random.split(random.PRNGKey(0))
input_dim = X.shape[1]
_, _, inverse_fun = transformation(rng, input_dim)

# %%
import jax
n_samples = 1000
rng, flow_rng = random.split(random.PRNGKey(0))
prior_samples = jax.random.normal(rng, (n_samples,2))
moon_samples = inverse_fun(params, prior_samples)[0]
plt.scatter(moon_samples[:, 0], moon_samples[:, 1])

# %%
# %%
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
import jax.numpy as jnp
from jax import random


# %%

def model(y_obs, params):
    dim = jnp.shape(y_obs)[0]
    lambdas = numpyro.sample("lambdas", dist.HalfCauchy(jnp.ones(dim)))
    tau = numpyro.sample("tau", dist.HalfCauchy(jnp.array([0.000025])))
    scales = lambdas*tau
    z = numpyro.sample("z", dist.Normal(jnp.zeros(dim), 1))  # Base dist
    x = numpyro.deterministic("x", inverse_fun(params, np.expand_dims(z, 0))[0].squeeze())  # Transform
    numpyro.sample("y", dist.Normal(x, scales), obs=y_obs)

# %%
y_obs = jnp.array([0, 0])

kernel = NUTS(model)
mcmc = MCMC(
    kernel,
    num_warmup=5000,
    num_samples=30000,
)

mcmc.run(random.PRNGKey(0), y_obs, params)
mcmc.print_summary()


# %% Plot posterior pairplot
import numpy as onp
import seaborn as sns
import pandas as pd
import numpy as onp
samples = mcmc.get_samples()

# %%
colnames = [f"{param}{i}" for param in samples.keys() for i in range(samples[param].shape[1])]
samples_array = onp.concatenate(list(samples.values()), axis = 1)
df = pd.DataFrame(samples_array, columns = colnames)


# %%
# plot og x, and y
df_og_x = pd.DataFrame(onp.array(X), columns=["x0", "x1"])
df_og_x["source"] = "sims"
df_denoised_x = df[["x0", "x1"]]
df_denoised_x["source"] = "denoised"

y_obs_onp = onp.array(y_obs)
obs_df = pd.DataFrame(
    [[y_obs_onp[0], y_obs_onp[1], "observation"]],
    columns = ["x0", "x1", "source"],
)

combined_df = pd.concat([df_og_x, df_denoised_x, obs_df])

# %%
combined_df = combined_df.reset_index(drop=True)
ax = sns.kdeplot(data=combined_df, x="x0", y="x1", hue="source", fill=True, common_norm = False, alpha=0.7)
plt.scatter(x=y_obs_onp[0], y=y_obs_onp[1], color="green")
ax.legend_.set_bbox_to_anchor((0.03, 0.35))
ax.legend_._set_loc(2)
# plt.savefig("two_moons_denoised.png")

# plt.savefig("two_moons_denoising.png")


# %%
sns.scatterplot(data=df, x = "x0", y="x1", alpha=0.1)

# %%
sns.kdeplot(data=df, x = "z0", y="z1", fill=True)

# %%
# %%
numpyro.render_model(model, model_args=(y_obs, params))



# %%  Pairplot with z
pp_samps = onp.concatenate(
    [onp.array(samples[k]) for k in ["z", "lambdas", "tau"]], axis=1
    )
colnames = ["z1", "z2", "lambda1", "lambda2", "tau"]

pp_samps = pd.DataFrame(pp_samps, columns = colnames)
sns.pairplot(
    pp_samps.sample(frac=1)[0:5000],
    vars = list(pp_samps),
    diag_kind="density",
    corner=True,
    plot_kws={"alpha": 0.1})


# %% pairplot with x
pp_samps = onp.concatenate(
    [onp.array(samples[k]) for k in ["x", "lambdas", "tau"]], axis=1
    )
colnames = ["x1", "x2", "lambda1", "lambda2", "tau"]

pp_samps = pd.DataFrame(pp_samps, columns = colnames)
sns.pairplot(
    pp_samps.sample(frac=1)[0:5000],
    vars = list(pp_samps),
    diag_kind="density",
    corner=True,
    plot_kws={"alpha": 0.01})



# %%

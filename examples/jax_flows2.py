# %%
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing
import jax.numpy as np
from jax import random
from rnpe.jax_flow import get_made_transformation, train_marginal_flow
from rnpe.denoise import horseshoe_denoiser_model
from numpyro.infer.hmc import NUTS
from numpyro.infer.mcmc import MCMC

# %% Toy data
n_samples = 10000
plot_range = [(-2, 2), (-2, 2)]
n_bins = 100
scaler = preprocessing.StandardScaler()
X, _ = datasets.make_moons(n_samples=n_samples, noise=.05)
X = scaler.fit_transform(X)
plt.hist2d(X[:, 0], X[:, 1], bins=n_bins, range=plot_range)[-1]
transformation = get_made_transformation(10)

# %%
key = random.PRNGKey(0)
flow, losses = train_marginal_flow(
    key, transformation, X, max_epochs=100, patience=5
    )

key, subkey = random.split(key)
samps = flow.sample(key, num_samples=1000)
samps = np.array(samps)
plt.plot(samps[:, 0], samps[:, 1], "r.")

# %%
y_obs = np.array([1.7, 1.5])
kernel = NUTS(horseshoe_denoiser_model)
mcmc = MCMC(
    kernel,
    num_warmup=5000,
    num_samples=30000
)

mcmc.run(random.PRNGKey(42), y_obs, flow)
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

# %%
sns.scatterplot(data=df, x = "x0", y="x1", alpha=0.1)

# %%
sns.scatterplot(data=df, x = "z0", y="z1")

# %%
from numpyro import render_model
render_model(horseshoe_denoiser_model, model_args=(y_obs,))

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

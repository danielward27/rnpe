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

# %%
key = random.PRNGKey(0)
transformation = get_made_transformation(10)
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
df_samples = pd.DataFrame(samples_array, columns = colnames)

# %%
sns.scatterplot(data=df_samples, x = "z0", y="z1", s=0.5)
plt.title("Inferred latents q(z|y)")

# %%
df_x = pd.DataFrame(onp.array(X), columns=["x0", "x1"])
df_x["source"] = "sims"
df_denoised_x = df_samples.loc[:, ["x0", "x1"]]
df_denoised_x["source"] = "denoised"

y_obs_onp = onp.array(y_obs)
df_obs = pd.DataFrame(
    [[y_obs_onp[0], y_obs_onp[1], "observation"]],
    columns = ["x0", "x1", "source"],
)

df_combined = pd.concat([df_x, df_denoised_x, df_obs])
df_combined = df_combined.reset_index(drop=True)

# %%
import seaborn as sns
sns.scatterplot(data=df_combined, x="x0", y="x1", hue="source", s=0.7)
plt.legend(loc='lower left')
plt.title("Inferred denoised samples")
plt.scatter(x=y_obs_onp[0], y=y_obs_onp[1], color="green")

# %%

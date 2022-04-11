from abc import ABC, abstractmethod
import os
import jax.numpy as jnp
import numpy as onp
from jax import random
import jax


class Task(ABC):
    @property
    def name(self):
        return type(self).__name__.lower()

    @abstractmethod
    def sample_prior(self, key: random.PRNGKey, n: int):
        "Draw n samples from the prior."
        pass

    @abstractmethod
    def simulate(self, key: random.PRNGKey, theta: jnp.ndarray):
        "Carry out simulations."
        pass

    @abstractmethod
    def generate_observation(self, key: random.PRNGKey):
        "Generate misspecified pseudo-observed data. Returns (theta_true, y)."
        pass

    def generate_dataset(self, key: random.PRNGKey, n: int, scale=True):
        "Generate optionally scaled dataset with pseudo-observed value and simulations"
        theta_key, x_key, obs_key = random.split(key, 3)
        theta = self.sample_prior(theta_key, n)
        x = self.simulate(x_key, theta)
        theta_true, y = self.generate_observation(obs_key)

        if scale:
            theta, theta_true, x, y = self.scale(theta, theta_true, x, y)

        data = {
            "theta": theta,
            "theta_true": theta_true,
            "x": x,
            "y": y,
        }

        return data

    @staticmethod
    def scale(theta, theta_true, x, y):
        "Center and scale data, using mean and std from theta and x."
        theta_means, theta_stds = theta.mean(axis=0), theta.std(axis=0)
        theta = (theta - theta_means) / theta_stds
        theta_true = (theta_true - theta_means) / theta_stds

        x_means, x_stds = x.mean(axis=0), x.std(axis=0)
        x = (x - x_means) / x_stds
        y = (y - x_means) / x_stds
        return theta, theta_true, x, y


class SIRSDE(Task):
    """Prior is uniform within triangle with vertices [(0,0), (0,0.5), (0.5,
    0.5)] such that beta > gamma. Note this example requires julia, and 
    may take a minute or two to compile."""

    def __init__(self, julia_env_path=".", misspecify_multiplier=0.95):
        self.julia_env_path = julia_env_path
        self.misspecify_multiplier = misspecify_multiplier
        self.x_names = [
            "log_mean",
            "log_median",
            "log_max_infections",
            "log_max_infections_day",
            "log_half_total_reached_at",
            "autocor_lag1",
        ]
        self.theta_names = ["beta", "gamma"]

    def sample_prior(self, key: random.PRNGKey, n: int):
        u1key, u2key = random.split(key)
        x = jnp.sqrt(random.uniform(u1key, (n,))) / 2
        y = (random.uniform(u2key, (n,)) * x) / 2
        return jnp.column_stack((x, y))

    def simulate(self, key: random.PRNGKey, theta: jnp.ndarray, summarise=True):
        "Simulates 365 days of sir model."
        key = key[1].item()
        # temporarily save parameters to file to be accessed from julia
        jl_script_f = os.path.dirname(os.path.realpath(__file__)) + "/julia/sirsde.jl"
        tmp_theta_f = f"_temp_theta_{key}.npz"
        tmp_x_f = f"_temp_x_{key}.npz"
        jnp.savez(tmp_theta_f, theta=theta)

        # Run julia simulation script, outputing to file
        try:
            command = f"""
            julia --project={self.julia_env_path} {jl_script_f} --seed={key}  \
                --theta_path={tmp_theta_f} --output_path={tmp_x_f}
            """
            os.system(command)
        finally:
            os.remove(tmp_theta_f)

        x = jnp.load(tmp_x_f)["x"]
        os.remove(tmp_x_f)

        if summarise:
            x = self.summarise(x)

        return x

    def summarise(self, x):
        @jax.jit
        @jax.vmap
        def autocorr_lag1(x):
            x1 = x[:-1]
            x2 = x[1:]
            x1_dif = x1 - x1.mean()
            x2_dif = x2 - x2.mean()
            numerator = (x1_dif * x2_dif).sum()
            denominator = jnp.sqrt((x1_dif ** 2).sum() * (x2_dif ** 2).sum())
            return numerator / denominator

        def cumulative_day(x, q):
            "Day when q proportion of total infections was reached."
            prop_i = (jnp.cumsum(x, axis=1).T / jnp.sum(x, axis=1)).T
            return jnp.argmax(prop_i > q, axis=1)

        summaries = [
            jnp.log(x.mean(axis=1)),
            jnp.log(jnp.median(x, axis=1)),
            jnp.log(jnp.argmax(x, axis=1) + 1),  # +1 incase 0 is max_day
            jnp.log(jnp.max(x, axis=1)),
            jnp.log(cumulative_day(x, 0.5)),
            autocorr_lag1(x),
        ]

        summaries = jnp.column_stack(summaries)
        return summaries

    def generate_observation(self, key: random.PRNGKey):
        theta_key, y_key = random.split(key)
        theta_true = self.sample_prior(theta_key, 1)
        y = self.simulate(y_key, theta_true, summarise=False)
        y = self.misspecify(y)
        y = self.summarise(y)
        return theta_true[0, :], y[0, :]

    def misspecify(self, x):
        x = onp.array(x)
        x = x.copy()
        sat_idx, sun_idx, mon_idx = [range(i, 365, 7) for i in range(1, 4)]
        sat_new = x[:, sat_idx] * self.misspecify_multiplier
        sun_new = x[:, sun_idx] * self.misspecify_multiplier
        missed_cases = (x[:, sat_idx] - sat_new) + (x[:, sun_idx] - sun_new)
        mon_new = x[:, mon_idx] + missed_cases
        for idx, new in zip([sat_idx, sun_idx, mon_idx], [sat_new, sun_new, mon_new]):
            x[:, idx] = new
        return jnp.array(x)


class FrazierGaussian(Task):
    """Task to infer mean of Gaussian using samples, with misspecified std.
    See https://arxiv.org/pdf/1708.01974.pdf."""

    def __init__(
        self,
        x_raw_dim: int = 100,
        prior_var: float = 25,
        likelihood_var: float = 1,
        misspecified_likeliood_var: float = 2,
    ):
        self.x_raw_dim = x_raw_dim
        self.prior_var = prior_var
        self.likelihood_var = likelihood_var
        self.misspecified_likelihood_var = misspecified_likeliood_var

    def sample_prior(self, key: random.PRNGKey, n: int):
        return random.normal(key, (n, 1)) * jnp.sqrt(self.prior_var)

    def simulate(self, key: random.PRNGKey, theta: jnp.ndarray):
        x_demean = random.normal(key, (theta.shape[0], self.x_raw_dim)) * jnp.sqrt(
            self.likelihood_var
        )
        x = x_demean + theta
        x = jnp.column_stack((x.mean(axis=1), x.var(axis=1)))
        return x

    def generate_observation(self, key: random.PRNGKey):
        theta_key, y_key = random.split(key)
        theta_true = self.sample_prior(theta_key, 1)
        y_demean = random.normal(
            y_key, (theta_true.shape[0], self.x_raw_dim)
        ) * jnp.sqrt(self.misspecified_likelihood_var)
        y = y_demean + theta_true
        y = jnp.column_stack((y.mean(axis=1), y.var(axis=1)))
        return theta_true[0, :], y[0, :]

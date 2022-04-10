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
        pass

    @abstractmethod
    def simulate(self, key: random.PRNGKey, theta: jnp.ndarray):
        pass

    @abstractmethod
    def misspecify(self, x):
        "Misspecification to apply to pseudo-observed data (prior to summarising)."
        pass

    @abstractmethod
    def summarise(self, x):
        pass

    def generate_dataset(self, key: random.PRNGKey, n: int, scale=True):
        "Generate scaled dataset with pseudo-observed value and simulations"
        theta_key, x_key = random.split(key)
        theta = self.sample_prior(theta_key, n + 1)
        x = self.simulate(x_key, theta, summarise=False)
        y, x = jax.numpy.split(x, jnp.array([1]), axis=0)
        theta_true, theta = jax.numpy.split(theta, jnp.array([1]), axis=0)

        y = self.misspecify(y)
        x = self.summarise(x)
        y = self.summarise(y)

        if scale:
            theta, theta_true, x, y = self.scale(theta, theta_true, x, y)

        data = {
            "theta": theta,
            "theta_true": jnp.squeeze(theta_true),
            "x": x,
            "y": jnp.squeeze(y),
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
    0.5)] such that beta > gamma. Note this example will take a minute or two to
    compile."""

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

    @staticmethod
    def summarise(x):
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

    def misspecify(self, x):
        x = onp.array(x)
        x = x.copy()
        days = ["saturday", "sunday", "monday"]
        sat_idx, sun_idx, mon_idx = [self.get_day_idx([d]) for d in days]
        mon_idx = mon_idx[1:]
        sat_new = x[:, sat_idx] * self.misspecify_multiplier
        sun_new = x[:, sun_idx] * self.misspecify_multiplier
        missed_cases = (x[:, sat_idx] - sat_new) + (x[:, sun_idx] - sun_new)
        mon_new = x[:, mon_idx] + missed_cases

        for idx, new in zip([sat_idx, sun_idx, mon_idx], [sat_new, sun_new, mon_new]):
            x[:, idx] = new
        return jnp.array(x)

    @staticmethod
    def get_day_idx(days: list):
        weekdays = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
        weekdays = {day: i for i, day in enumerate(weekdays)}
        idxs = []
        for day in days:
            idxs += list(range(weekdays[day], 365, 7))
        return sorted(idxs)


from abc import ABC, abstractmethod
import os
import jax.numpy as jnp
import numpy as onp
from jax import random
import jax
import numba


class Task(ABC):
    @property
    def name(self):
        return type(self).__name__

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
        x = remove_nans_and_warn(x)
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

    def scale(self, theta, theta_true, x, y):
        "Center and scale data, using mean and std from theta and x."
        theta_means, theta_stds = theta.mean(axis=0), theta.std(axis=0)
        theta = (theta - theta_means) / theta_stds
        theta_true = (theta_true - theta_means) / theta_stds

        x_means, x_stds = x.mean(axis=0), x.std(axis=0)
        x = (x - x_means) / x_stds
        y = (y - x_means) / x_stds

        self.scales = {
            "theta_mean": theta_means,
            "theta_std": theta_stds,
            "x_mean": x_means,
            "x_std": x_stds,
        }
        return theta, theta_true, x, y


def remove_nans_and_warn(x):
    nan_rows = jnp.any(jnp.isnan(x), axis=1)
    n_nan = nan_rows.sum()
    if n_nan > 0:
        x = x[~nan_rows]
        print(f"Warning {n_nan} simulations contained NAN values have been removed.")
    return x


class SIR(Task):
    """Prior is uniform [0, 0.5] with constraint such that beta > gamma. Note
    this example requires julia, and may take a minute or two to compile."""

    def __init__(self, julia_env_path=".", misspecify_multiplier=0.95):
        self.julia_env_path = julia_env_path
        self.misspecify_multiplier = misspecify_multiplier
        self.x_names = [
            "Mean",     # Mean infections
            "Median",   # Median infections
            "Max",      # Max infections
            "Max Day",  # Day of max infections
            "Half Day", # Day where half of cumulative total number of infections reached
            "Autocor",  # Autocorrelation lag 1
        ]
        self.theta_names = [r"$\beta$", r"$\gamma$"]

    def sample_prior(self, key: random.PRNGKey, n: int):
        u1key, u2key = random.split(key)
        x = jnp.sqrt(random.uniform(u1key, (n,))) / 2
        y = random.uniform(u2key, (n,)) * x
        return jnp.column_stack((x, y))

    def simulate(self, key: random.PRNGKey, theta: jnp.ndarray, summarise=True):
        "Simulates 365 days of sir model."
        key = key[1].item()
        # temporarily save parameters to file to be accessed from julia
        jl_script_f = os.path.dirname(os.path.realpath(__file__)) + "/julia/SIR.jl"
        tmp_theta_f = f"_temp_theta_{key}.npz"
        tmp_x_f = f"_temp_x_{key}.npz"
        jnp.savez(tmp_theta_f, theta=theta)

        try:
            # Run julia simulation script, outputing to file
            command = f"""
            julia --project={self.julia_env_path} {jl_script_f} --seed={key}  \
                --theta_path={tmp_theta_f} --output_path={tmp_x_f}
            """
            os.system(command)
            x = jnp.load(tmp_x_f)["x"]
        finally:
            for f in [tmp_theta_f, tmp_x_f]:
                try:
                    os.remove(f)
                except OSError:
                    pass

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
            jnp.log(jnp.max(x, axis=1)),
            jnp.log(jnp.argmax(x, axis=1) + 1),  # +1 incase 0 is max_day
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


class Gaussian(Task):
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
        self.theta_names = [r"$\mu$"]
        self.x_names = [r"$\mu$", r"$\sigma^2$"]

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


class CS(Task):
    def __init__(self):
        self.theta_names = [r"$\lambda_c$", r"$\lambda_p$", r"$\lambda_d$"]
        self.x_names = [
            "N Stromal",
            "N Cancer",
            "Mean Min Dist",
            "Max Min Dist",
        ]

    def simulate(
        self,
        key: random.PRNGKey,
        theta: jnp.array,
        summarise: bool = True,
        necrosis: bool = False,
    ):
        theta = onp.array(theta)
        onp.random.seed(key[1].item())

        x = []
        for row in theta:
            xi = self._simulate(
                cell_rate=row[0],
                cancer_parent_rate=row[1],
                cancer_daughter_rate=row[2],
                necrosis=necrosis,
            )
            if summarise is True:
                xi = self.summarise(*xi)
            x.append(xi)

        if summarise:
            x = jnp.row_stack(x)

        return x

    def _simulate(
        self,
        cell_rate: float = 1000,
        cancer_parent_rate: float = 5,
        cancer_daughter_rate: float = 30,
        necrosis=False,
    ):
        num_cells = onp.random.poisson(cell_rate)
        cells = onp.random.uniform(size=(num_cells, 2))

        num_cancer_parents = onp.random.poisson(cancer_parent_rate) + 1
        cancer_parents = onp.random.uniform(0, 1, size=(num_cancer_parents, 2))

        num_cancer_daughters = (
            onp.random.poisson(cancer_daughter_rate, (num_cancer_parents,)) + 1
        )

        dists = dists_between(cells, cancer_parents)  # n_cells by n_parents
        radii = self.n_nearest_dists(dists, num_cancer_daughters)
        is_cancer = (dists <= radii).any(axis=1)

        if necrosis:
            has_necrosis = onp.random.binomial(1, p=0.75, size=(num_cancer_parents,))
            has_necrosis = has_necrosis.astype(bool)
            if has_necrosis.sum() > 0:
                bl_array = dists[:, has_necrosis] < (radii[has_necrosis] * 0.8)
                necrotized = onp.any(bl_array, axis=1)
                cells, is_cancer = (
                    cells[~necrotized],
                    is_cancer[~necrotized],
                )

        return cells, is_cancer

    def summarise(self, cells, is_cancer, threshold_n_stromal: int = 50):
        """Calculate summary statistics, threshold_n_stromal limits the number
        of stromal cells (trade off for efficiency)."""
        num_cancer = is_cancer.sum()

        if num_cancer == is_cancer.shape[0]:
            print("Warning, no stromal cells. Returning nan for summary statistics.")
            return jnp.full(len(self.x_names), jnp.nan)

        num_stromal = (~is_cancer).sum()
        threshold_num_stromal = min(threshold_n_stromal, num_stromal)
        cancer = cells[is_cancer]
        stromal = cells[~is_cancer][:threshold_num_stromal]

        dists = dists_between(stromal, cancer)

        min_dists = dists.min(axis=1)
        mean_nearest_cancer = min_dists.mean()
        max_nearest_cancer = min_dists.max()

        summaries = [
            num_stromal,
            num_cancer,
            mean_nearest_cancer,
            max_nearest_cancer,
        ]

        return jnp.array(summaries)

    def generate_observation(self, key: random.PRNGKey):
        theta_key, y_key = random.split(key)
        theta_true = self.sample_prior(theta_key, 1)
        y = self.simulate(y_key, theta_true, necrosis=True)
        return jnp.squeeze(theta_true), jnp.squeeze(y)

    def sample_prior(self, key: random.PRNGKey, n: int):
        keys = random.split(key, 3)
        cell_rate = random.uniform(keys[0], (n,), minval=200, maxval=1500)
        cancer_parent_rate = random.uniform(keys[1], (n,), minval=3, maxval=20)
        cancer_daughter_rate = random.uniform(keys[2], (n,), minval=10, maxval=20)
        return jnp.column_stack([cell_rate, cancer_parent_rate, cancer_daughter_rate])

    def n_nearest_dists(self, dists, n_points):
        "Minimum distance containing n points. n_points to match axis dists in axis 1."
        assert dists.shape[1] == len(n_points)
        d_sorted = onp.partition(dists, kth=n_points, axis=0)
        min_d = d_sorted[n_points, onp.arange(dists.shape[1])]
        return min_d

@numba.njit(fastmath=True)
def dists_between(a, b):
    "Returns a.shape[0] by b.shape[0] l2 norms between rows of arrays."
    dists = []
    for ai in a:
        for bi in b:
            dists.append(onp.linalg.norm(ai - bi))
    return onp.array(dists).reshape(a.shape[0], b.shape[0])
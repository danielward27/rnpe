# %%
import jax
import jax.numpy as jnp
from flowjax.flows import Flow
from jax.scipy.special import logsumexp
from tqdm import tqdm


# Too memory intensive alone
# def robust_posterior_log_prob(flow: Flow, theta: jnp.ndarray, denoised: jnp.ndarray):
#     """Given a flow q(theta|x), a matrix of theta, and denoised observations,
#     evaluate the posterior probability as the expectation over denoised samples."""
#     f = jax.jit(jax.vmap(flow.log_prob, in_axes=(None, 0)))
#     res = f(theta, denoised)
#     return logsumexp(res - jnp.log(denoised.shape[0]), axis=0)


def robust_posterior_log_prob(
    flow: Flow,
    theta: jnp.ndarray,
    denoised: jnp.ndarray,
    thin_denoised: int = 1,
    show_progress=False,
):
    """Given a flow q(theta|x), a matrix of theta, and denoised observations,
    evaluate the posterior probability p(theta|y) as the expectation over
    denoised samples."""

    @jax.jit
    def _inner(theta_row, denoised):
        probs = jax.vmap(flow.log_prob, in_axes=(None, 0))(theta_row, denoised)
        return logsumexp(probs - jnp.log(denoised.shape[0]))

    probs = []
    loop = tqdm(theta) if show_progress else theta
    for theta_row in loop:
        probs.append(_inner(theta_row, denoised[::thin_denoised]))

    return jnp.array(probs)


def log_prob_theta_true(flow, theta_true, denoised, y):
    "log prob theta*, returns tuple (robust, nonrobust)."
    theta_2d = jnp.expand_dims(theta_true, 0)
    robust_lp_theta_true = robust_posterior_log_prob(flow, theta_2d, denoised)
    naive_lp_theta_true = flow.log_prob(theta_true, y)
    return robust_lp_theta_true.item(), naive_lp_theta_true.item()


def highest_posterior_density(
    flow: Flow,
    theta_true: jnp.ndarray,
    denoised: jnp.ndarray,
    robust_samples: jnp.ndarray,
    naive_samples: jnp.ndarray,
    y: jnp.ndarray,
    thin_denoised: int = 1,
    show_progress: bool = True,
):
    """Monte carlo approximation to find the smallest heighest density region
    that includes the true parameter. Returns a tuple of 100*(1-alpha)% values
    (robust, non_robust).
    """
    robust_lps = robust_posterior_log_prob(
        flow, robust_samples, denoised, thin_denoised, show_progress
    )
    naive_lps = flow.log_prob(naive_samples, y)

    robust_lp_theta_true, naive_lp_theta_true = log_prob_theta_true(
        flow, theta_true, denoised, y
    )

    robust_hpd = jnp.mean(robust_lps > robust_lp_theta_true).item()
    naive_hpd = jnp.mean(naive_lps > naive_lp_theta_true).item()
    return 100 * robust_hpd, 100 * naive_hpd


def calculate_metrics(
    flow: Flow,
    theta_true: jnp.ndarray,
    denoised: jnp.ndarray,
    robust_samples: jnp.ndarray,
    naive_samples: jnp.ndarray,
    y: jnp.ndarray,
):
    """Calculate metrics (log_prob_theta_true and highest_posterior_density),
    for robust and non-robust approaches. returns dictionary."""

    theta_true_lp_robust, theta_true_lp_naive = log_prob_theta_true(
        flow, theta_true, denoised, y
    )
    hpd_robust, hpd_naive = highest_posterior_density(
        flow=flow,
        theta_true=theta_true,
        denoised=denoised,
        robust_samples=robust_samples,
        naive_samples=naive_samples,
        y=y,
    )

    metrics = {
        "with error model": {
            "prob_theta*": theta_true_lp_robust,
            "hpd": hpd_robust,
            "point_estimate_residuals": robust_samples.mean(axis=0) - theta_true,
        },
        "without error model": {
            "prob_theta*": theta_true_lp_naive,
            "hpd": hpd_naive,
            "point_estimate_residuals": naive_samples.mean(axis=0) - theta_true,
        },
    }
    return metrics

import jax
import jax.numpy as jnp
from jax import random
from flowjax.distributions import Distribution as FlowJaxDist
from jax.scipy.special import logsumexp
from tqdm import tqdm
from rnpe.tasks import Task
from sbibm.metrics import c2st
import torch
import numpy as onp


# Too memory intensive
# def robust_posterior_log_prob(flow: FlowJaxDist, theta: jnp.ndarray, denoised: jnp.ndarray):
#     """Given a flow q(theta|x), a matrix of theta, and denoised observations,
#     evaluate the posterior probability as the expectation over denoised samples."""
#     f = jax.jit(jax.vmap(flow.log_prob, in_axes=(None, 0)))
#     res = f(theta, denoised)
#     return logsumexp(res - jnp.log(denoised.shape[0]), axis=0)

def robust_posterior_log_prob(
    flow: FlowJaxDist, theta: jnp.ndarray, denoised: jnp.ndarray, show_progress=False,
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
        probs.append(_inner(theta_row, denoised))

    return jnp.array(probs)

def calculate_metrics(
    key: random.PRNGKey,
    task: Task,
    flow: FlowJaxDist,
    noisy_flow: FlowJaxDist,
    theta_true: jnp.ndarray,
    denoised: jnp.ndarray,
    robust_samples: jnp.ndarray,
    naive_samples: jnp.ndarray,
    noisy_samples: jnp.ndarray,
    y_obs: jnp.ndarray,
    thin_denoised_hpd: int = 20,
    show_progress: bool = True,
):
    """Calculate metrics for robust-NPE and NPE.

    Args:
        flow (FlowJaxDist): Flow approximating p(theta|x)
        theta_true (jnp.ndarray): True parameters associated with y.
        denoised (jnp.ndarray): Denoised observations.
        robust_samples (jnp.ndarray): Robust NPE posterior samples
        naive_samples (jnp.ndarray): NPE posterior samples
        y (jnp.ndarray): Observation.
        thin_denoised_hpd (int, optional): Thinning of denoised samples when calculating
            HPD (for computational reasons). Defaults to 1.
        show_progress (bool): Whether to show progress bar for computing HPD.

    Returns:
        dict: Contains the log probability of the true parameters, the HPD%, and residuals
            based on posterior means as a point esimate, for both Robust NPE and NPE
    """
    samples = {"RNPE": robust_samples, "NPE": naive_samples, "NNPE": noisy_samples}
    metrics = {k: {} for k in samples.keys()}

    # C2ST if referece posterior available
    if task.tractable_posterior:
        key, subkey = random.split(key)
        scaled_y_obs = y_obs*task.scales["x_std"] + task.scales["x_mean"]
        true_posterior_samples = task.get_true_posterior_samples(subkey, scaled_y_obs, robust_samples.shape[0])
        true_posterior_samples = (true_posterior_samples - task.scales["theta_mean"]) / task.scales["theta_std"]
        
        for key, samps in samples.items():
            acc = c2st(torch.from_numpy(onp.array(true_posterior_samples)), torch.from_numpy(onp.array(samps)), n_folds=3)
            metrics[key]["C2ST"] = acc

    else:
        for v in metrics.values():
            v["C2ST"] = None
    
    # lptheta*
    metrics["RNPE"]["log_prob_theta*"] = robust_posterior_log_prob(flow, theta_true[None, :], denoised).item()
    metrics["NPE"]["log_prob_theta*"] = flow.log_prob(theta_true, y_obs).item()
    metrics["NNPE"]["log_prob_theta*"] = noisy_flow.log_prob(theta_true, y_obs).item()
    
    # HPD
    robust_lps = robust_posterior_log_prob(flow, robust_samples, denoised[::thin_denoised_hpd], show_progress)
    naive_lps = flow.log_prob(naive_samples, y_obs)
    noisy_lps = noisy_flow.log_prob(noisy_samples, y_obs)

    for k, lps in zip(["RNPE", "NPE", "NNPE"], [robust_lps, naive_lps, noisy_lps]):
        metrics[k]["hpd"] = jnp.mean(lps > metrics[k]["log_prob_theta*"]).item() * 100
    
    # Point estimate residuals
    for k, samps in samples.items():
        metrics[k]["point_estimate_residuals"] = samps.mean(axis=0) - theta_true

    return metrics

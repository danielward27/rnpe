from flowjax.flows import block_neural_autoregressive_flow, coupling_flow
from flowjax.bijections.transformers import RationalQuadraticSplineTransformer
from flowjax.distributions import Normal
from flowjax.train_utils import train_flow
import jax.numpy as jnp
from jax import random
from numpyro.infer import MCMC, HMC, MixedHMC, init_to_value
from rnpe.denoise import spike_and_slab_denoiser, spike_and_slab_denoiser_hyperprior
from rnpe.tasks import SIR, Gaussian, GaussianLinear, CS
from rnpe.metrics import calculate_metrics
from time import time
import pickle
import argparse
import os

# Example command from project root directory:
# python -m scripts.run_task --task-name="Gaussian" --seed=0


class Timer:
    "Basic timer to record and print run times."

    results: dict = {}
    label: str = ""

    def start(self, label: str):
        print(f"Starting {label}")
        self.label = label
        self.start_time = time()

    def stop(self):
        elapsed = time() - self.start_time
        print(f"{self.label}: {elapsed:.2f}s")
        self.results[self.label] = elapsed


def rescale_results(res):
    x_mean, x_std = res["scales"]["x_mean"], res["scales"]["x_std"]
    theta_mean, theta_std = res["scales"]["theta_mean"], res["scales"]["theta_std"]

    res["data"]["x"] = res["data"]["x"] * x_std + x_mean
    res["data"]["y"] = res["data"]["y"] * x_std + x_mean
    res["data"]["theta"] = res["data"]["theta"] * theta_std + theta_mean
    res["data"]["theta_true"] = res["data"]["theta_true"] * theta_std + theta_mean
    res["mcmc_samples"]["x"] = res["mcmc_samples"]["x"] * x_std + x_mean

    res["posterior_samples"]["NPE"] = (
        res["posterior_samples"]["NPE"] * theta_std + theta_mean
    )
    res["posterior_samples"]["RNPE"] = (
        res["posterior_samples"]["RNPE"] * theta_std + theta_mean
    )
    return res


def add_spike_and_slab_error(
    key: random.PRNGKey, x: jnp.ndarray, slab_scale: float, spike_scale: float = 0.01
):
    keys = random.split(key, 3)
    misspecified = random.bernoulli(keys[0], shape=x.shape)
    spike = random.normal(keys[2], shape=x.shape) * spike_scale
    slab = random.cauchy(keys[1], shape=x.shape) * slab_scale
    return x + misspecified * slab + (1 - misspecified) * spike


def main(args):

    if args.misspecification_hyperprior:
        denoiser = spike_and_slab_denoiser_hyperprior
    else:
        denoiser = spike_and_slab_denoiser

    misspecified = not args.well_specified

    #### Carry out simulations ####
    tasks = {
        "SIR": SIR,
        "Gaussian": Gaussian,
        "CS": CS,
        "GaussianLinear": GaussianLinear
    }
    task = tasks[args.task_name]()
    key, subkey = random.split(random.PRNGKey(args.seed))

    timer = Timer()
    timer.start("simulations")
    data = task.generate_dataset(subkey, args.n_sim, misspecified=misspecified)
    timer.stop()

    #### Robust inference ####
    # Train marginal likelihood flow
    key, flow_key, train_key = random.split(key, 3)
    base_dist = Normal(data["x"].shape[1])
    x_flow = block_neural_autoregressive_flow(flow_key, base_dist)

    timer.start("q(x)_training")
    x_flow, x_losses = train_flow(
        train_key,
        x_flow,
        data["x"],
        learning_rate=0.01,
        max_epochs=args.max_epochs,
        show_progress=args.show_progress,
    )
    timer.stop()

    # Denoise observation with MCMC
    init = init_to_value(
        values={"x": data["x"][0], "misspecified": jnp.ones(len(data["y"]), int)}
    )

    kernel = MixedHMC(
        HMC(denoiser, trajectory_length=1, init_strategy=init, target_accept_prob=0.95,)
    )

    mcmc = MCMC(
        kernel,
        num_warmup=args.mcmc_warmup,
        num_samples=args.mcmc_samples,
        progress_bar=args.show_progress,
    )

    key, mcmc_key = random.split(key)
    model_kwargs = {"y_obs": data["y"], "flow": x_flow, "slab_scale": args.slab_scale}

    timer.start("q(x|y)_sampling")
    mcmc.run(mcmc_key, **model_kwargs)
    timer.stop()

    # Carry out posterior inference
    key, flow_key, train_key = random.split(key, 3)
    base_dist = Normal(data["theta"].shape[1])
    transformer = RationalQuadraticSplineTransformer(K=10, B=5)
    posterior_flow = coupling_flow(flow_key, base_dist, transformer, cond_dim=data["x"].shape[1])

    timer.start("q(theta|x)_training")
    posterior_flow, npe_losses = train_flow(
        train_key,
        posterior_flow,
        data["theta"],
        data["x"],
        max_epochs=args.max_epochs,
        learning_rate=0.0005,
        show_progress=args.show_progress,
    )
    timer.stop()

    timer.start("q(theta|y)_training")
    key, subkey = random.split(key)
    noisy_sims = add_spike_and_slab_error(key, data["x"], args.slab_scale)

    noisy_posterior_flow, npe_losses = train_flow(
        train_key,
        posterior_flow,
        data["theta"],
        noisy_sims,
        max_epochs=args.max_epochs,
        learning_rate=0.0005,
        show_progress=args.show_progress,
    )
    timer.stop()

    timer.start("Sample posteriors and calculate metrics")

    #### Sample posteriors ####
    denoised = mcmc.get_samples()["x"]
    key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)
    denoised_subset = random.permutation(subkey1, denoised)[: args.posterior_samples]
    robust_npe_samples = posterior_flow.sample(subkey2, denoised_subset)
    naive_npe_samples = posterior_flow.sample(
        subkey3, data["y"], args.posterior_samples
    )
    noisy_npe_samples = noisy_posterior_flow.sample(
        subkey4, data["y"], args.posterior_samples
    )

    #### Calculate metrics ####
    key, subkey = random.split(key)

    metrics = calculate_metrics(
        key=key,
        task=task,
        flow=posterior_flow,
        noisy_flow=noisy_posterior_flow,
        theta_true=data["theta_true"],
        denoised=denoised,
        robust_samples=robust_npe_samples,
        naive_samples=naive_npe_samples,
        noisy_samples=noisy_npe_samples,
        y_obs=data["y"],
        show_progress=args.show_progress,
    )

    timer.stop()

    results = {
        "data": data,
        "mcmc_samples": mcmc.get_samples(),
        "metrics": metrics,
        "posterior_samples": {
            "RNPE": robust_npe_samples,
            "NPE": naive_npe_samples,
            "NNPE": noisy_npe_samples,
        },
        "runtimes": timer.results,
        "names": {"x": task.x_names, "theta": task.theta_names},
        "scales": task.scales,
        "losses": {"x": x_losses, "theta|x": npe_losses},
    }

    results = rescale_results(results)

    fname = f"{args.results_dir}/seed={args.seed}_slab_scale={args.slab_scale}_hyperprior={args.misspecification_hyperprior}_misspecified={misspecified}.pickle"

    with open(fname, "wb",) as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNPE")
    parser.add_argument("--task-name", type=str, help="Gaussian, GaussianLinear, SIR or CS")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--results-dir", help="defaults to results/task_name")
    parser.add_argument("--n-sim", default=50000, type=int)
    parser.add_argument("--max-epochs", default=50, type=int)
    parser.add_argument("--mcmc-warmup", default=20000, type=int)
    parser.add_argument("--mcmc-samples", default=100000, type=int)
    parser.add_argument("--posterior-samples", default=10000, type=int)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument(
        "--well-specified",
        action="store_true",
        help="whether to use a misspecified or well specified observation",
    )
    parser.add_argument("--misspecification-hyperprior", action="store_true")
    parser.add_argument("--slab-scale", default=0.25, type=float)

    args = parser.parse_args()

    assert args.posterior_samples <= args.mcmc_samples

    if args.results_dir is None:
        args.results_dir = f"results/{args.task_name}"

    assert os.path.isdir(args.results_dir)

    main(args)


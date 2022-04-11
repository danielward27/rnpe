from flowjax.flows import BlockNeuralAutoregressiveFlow, NeuralSplineFlow
from flowjax.train_utils import train_flow
import jax.numpy as jnp
from jax import random
from numpyro.infer import MCMC, HMC, MixedHMC, init_to_value
from rnpe.denoise import spike_and_slab_denoiser
from rnpe.tasks import SIRSDE, FrazierGaussian
from time import time
from jax.scipy.special import logsumexp
import pickle
import argparse
import os

# Example command from project root directory:
# python -m scripts.run_task --task-name="fraziergaussian" --seed=1


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


def main(args):

    # Carry out simulations
    tasks = {
        "sirsde": SIRSDE,
        "fraziergaussian": FrazierGaussian,
    }
    task = tasks[args.task_name]()
    key, subkey = random.split(random.PRNGKey(args.seed))

    timer = Timer()
    timer.start("simulations")
    data = task.generate_dataset(subkey, args.n_sim)
    timer.stop()

    # Train marginal likelihood flow
    key, flow_key, train_key = random.split(key, 3)
    x_flow = BlockNeuralAutoregressiveFlow(
        flow_key, target_dim=data["x"].shape[1], block_size=(8, 8)
    )

    timer.start("q(x)_training")
    x_flow, _ = train_flow(
        train_key, x_flow, data["x"], learning_rate=0.01, max_epochs=args.max_epochs
    )
    timer.stop()

    # Denoise observation with MCMC
    init = init_to_value(
        values={"x": data["x"][0], "misspecified": jnp.ones(len(data["y"]), int)}
    )

    kernel = MixedHMC(
        HMC(
            spike_and_slab_denoiser,
            trajectory_length=1,
            init_strategy=init,
            target_accept_prob=0.95,
        )
    )

    mcmc = MCMC(
        kernel,
        num_warmup=args.mcmc_warmup,
        num_samples=args.mcmc_samples,
        progress_bar=False,
    )

    key, mcmc_key = random.split(key)
    model_args = [data["y"], x_flow]

    timer.start("q(x|y)_sampling")
    mcmc.run(mcmc_key, *model_args)
    timer.stop()

    # Carry out posterior inference
    key, flow_key, train_key = random.split(key, 3)

    posterior_flow = NeuralSplineFlow(
        flow_key, target_dim=data["theta"].shape[1], condition_dim=data["x"].shape[1]
    )

    timer.start("q(theta|x)_training")
    posterior_flow, _ = train_flow(
        train_key,
        posterior_flow,
        data["theta"],
        data["x"],
        max_epochs=args.max_epochs,
        learning_rate=0.0005,
    )
    timer.stop()

    denoised = mcmc.get_samples()["x"]

    # Calculate log probabilities of the true parameters with and without error model.
    non_robust_lp = posterior_flow.log_prob(data["theta_true"], data["y"])

    # Robust posterior as expectation w.r.t. q(x|y)
    rep_theta_true = jnp.broadcast_to(
        data["theta_true"], (denoised.shape[0], len(data["theta_true"]))
    )
    robust_lps = posterior_flow.log_prob(rep_theta_true, denoised)
    robust_lp = logsumexp(robust_lps - jnp.log(args.mcmc_samples))

    results = {
        "no_error_model_logprob_theta_true": non_robust_lp.item(),
        "error_model_logprob_theta_true": robust_lp.item(),
    }

    time_results = {k + "_time": v for k, v in timer.results.items()}

    results = results | time_results

    with open(f"{args.results_dir}/{args.seed}.pickle", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust NPE")
    parser.add_argument("--task-name", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--results-dir", help="String. If not given, will use")
    parser.add_argument("--n-sim", default=50000, type=int)
    parser.add_argument("--max-epochs", default=50, type=int)
    parser.add_argument("--mcmc-warmup", default=20000, type=int)
    parser.add_argument("--mcmc-samples", default=100000, type=int)
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = f"results/{args.task_name}"

    assert os.path.isdir(args.results_dir)

    main(args)

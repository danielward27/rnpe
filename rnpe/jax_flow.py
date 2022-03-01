# %%
from collections import namedtuple
from functools import partial
from typing import Callable
import jax.numpy as np
from jax import grad, jit, random
import optax
import flows
from rnpe.utils.jax_flows import masked_transform, count_fruitless
from rnpe.utils.train_val_split import jax_train_val_split

from tqdm import tqdm

def get_made_transformation(  # TODO add option for hidden dim size/layers!
    flow_layers : int = 5
    ):
    transformation = flows.Serial(*(flows.MADE(masked_transform), flows.Reverse()) * flow_layers) 
    return transformation


def train_marginal_flow(
    key : random.PRNGKey,
    transformation : Callable,
    x : np.DeviceArray,
    max_epochs : int,
    patience : int = 5,
    batch_size : int = 128,
    lr : float = 1e-4,
    val_prop : float = 0.1,
    show_progress: bool = True,
    ):
    assert x.ndim == 2
    input_dim = x.shape[1]
    key, sub_key = random.split(key)
    _, _, inverse_fun = transformation(key, input_dim) 

    flow = flows.Flow(
        transformation,
        flows.Normal(),
    )

    key, sub_key = random.split(key)
    params, log_prob, sample = flow(key, input_dim)
    loss = lambda params, x : -log_prob(params, x).mean()
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    epochs = range(max_epochs)

    def loss(params, x):
        return -log_prob(params, x).mean()

    @jit
    def step(params, opt_state, x):
        grads = grad(loss)(params, x)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    key, sub_key = random.split(key)
    train_x, val_x = jax_train_val_split(key, x, val_prop)

    pbar = tqdm(epochs) if show_progress else epochs

    losses = {"train": [], "val": []}
    for epoch in pbar:
        key, sub_key = random.split(key)
        train_x = random.permutation(key, train_x)
        batches = range(0, len(train_x), batch_size)

        for batch_index in batches:
            params, opt_state = step(
                params,
                opt_state,
                train_x[batch_index:batch_index+batch_size]
                )

        train_loss = loss(params, train_x).item()
        losses["train"].append(train_loss)

        val_loss = loss(params, val_x).item()
        losses["val"].append(val_loss)

        if val_loss == min(losses["val"]):
            best_params = params

        elif count_fruitless(losses["val"]) >= patience:
            print("Maximum patience reached.")
            break

        

    Flow = namedtuple("Flow", "log_prob sample transform_noise")
    flow = Flow(
        *[partial(f, params=best_params) for f in [log_prob, sample, inverse_fun]]
        )
    return  flow, losses








# %%

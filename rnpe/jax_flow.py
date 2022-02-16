# %%
from collections import namedtuple
from functools import partial
from typing import Callable
import jax.numpy as np
from jax import grad, value_and_grad, jit, random
from jax.experimental import optimizers
import flows
from rnpe.jax_flow_utils import masked_transform
from rnpe.jax_flow_utils import train_val_split, count_fruitless_epochs
from tqdm import tqdm

def get_made_transformation(  # TODO add option for hidden dim size/layers!
    flow_layers : int = 5
    ):
    transformation = flows.Serial(*(flows.MADE(masked_transform), flows.Reverse()) * flow_layers)  # TODO shuffle instead of reverse?  
    return transformation


def train_marginal_flow(
    key : random.PRNGKey,
    transformation : Callable,
    x : np.DeviceArray,
    max_epochs : int,
    patience : int = 5,
    batch_size : int = 128,
    lr : float = 1e-4,
    train_prop : float = 0.9,
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
    opt_init, opt_update, get_params = optimizers.adam(step_size=lr)
    opt_state = opt_init(params)
    epochs = range(max_epochs)

    def loss(params, x):
        return -log_prob(params, x).mean()

    @jit
    def train_step(i, opt_state, x_train):
        params = get_params(opt_state)
        gradients = grad(loss)(params, x_train)
        return opt_update(i, gradients, opt_state)

    key, sub_key = random.split(key)
    train_x, val_x = train_val_split(key, x, train_prop)

    # TODO get train and test loss...
    step_idx=0
    pbar = tqdm(epochs) if show_progress else epochs

    losses = {"train": [], "val": []}
    for epoch in pbar:
        key, sub_key = random.split(key)
        train_x = random.permutation(key, train_x)
        batches = range(0, len(train_x), batch_size)
        for batch_index in batches:
            opt_state = train_step(
                step_idx,
                opt_state,
                train_x[batch_index:batch_index+batch_size]
                )
            step_idx +=1

        params = get_params(opt_state)

        train_loss = loss(params, train_x).item()
        losses["train"].append(train_loss)

        val_loss = loss(params, val_x).item()
        losses["val"].append(val_loss)

        if val_loss == min(losses["val"]):
            best_params = params

        elif count_fruitless_epochs(losses["val"]) >= patience:
            print("Maximum patience reached.")
            break

    FlowTuple = namedtuple("Flow", "log_prob sample transform_noise")
    flow = FlowTuple(
        *[partial(f, params=best_params) for f in [log_prob, sample, inverse_fun]]
        )
    return  flow, losses








# %%

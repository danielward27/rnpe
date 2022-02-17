import jax.numpy as np
from jax import random
from jax.example_libraries import stax
import flows

def get_masks(input_dim, hidden_dim=64, num_hidden=2):
    masks = []
    input_degrees = np.arange(input_dim)
    degrees = [input_degrees]

    for _ in range(num_hidden + 1):
        degrees += [np.arange(hidden_dim) % (input_dim - 1)]
    degrees += [input_degrees % input_dim - 1]

    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [np.transpose(np.expand_dims(d1, -1) >= np.expand_dims(d0, 0)).astype(np.float32)]
    return masks

def masked_transform(rng, input_dim): # TODO why num hidden 1 here but 2 above?
    masks = get_masks(input_dim, hidden_dim=64, num_hidden=1)
    init_fun, apply_fun = stax.serial(
        flows.MaskedDense(masks[0]),
        stax.Relu,
        flows.MaskedDense(masks[1]),
        stax.Relu,
        flows.MaskedDense(masks[2].tile(2)),
    )
    _, params = init_fun(rng, (input_dim,))
    return params, apply_fun


def train_val_split(
    key: random.PRNGKey,
    x: np.DeviceArray,
    train_prop: float = 0.9):
    assert 0 <= train_prop <= 1
    x = random.permutation(key, x)
    n_train = round(train_prop*x.shape[0])
    return x[0:n_train], x[n_train:]


def count_fruitless_epochs(losses: list):
    """Given a list of losses from each epoch, count the number of epochs since
    the minimum loss"""
    min_idx = np.array(losses).argmin().item()
    return len(losses) - min_idx - 1
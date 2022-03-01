import torch
from torch.utils.data import Dataset

import jax.numpy as jnp
import jax.random as random

def torch_train_val_split_dataset(dataset: Dataset, val_prop=0.1):
    """Perform train-val split on torch dataset, returning tuple (train_dataset,
    val_dataset)"""
    n = len(dataset)
    val_size = round(n * val_prop)
    train_size = n - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    return train_dataset, val_dataset


def jax_train_val_split(
    key: random.PRNGKey,
    x: jnp.ndarray,
    val_prop: float = 0.1):
    assert 0 <= val_prop <= 1
    x = random.permutation(key, x)
    n_val = round(val_prop*x.shape[0])
    return x[:-n_val], x[n_val:]
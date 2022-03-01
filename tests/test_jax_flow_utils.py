import pytest
from jax import random
import jax.numpy as np
from rnpe.jax_flow_utils import train_val_split, count_fruitless

def test_train_val_split():
    key = random.PRNGKey(0)
    x = random.normal(key, (100, 3))
    train_x, val_x = train_val_split(key, x, 0.9)
    assert train_x.shape == (90, 3)
    assert val_x.shape == (10, 3)

    train_x, val_x = train_val_split(key, x, 1)
    assert val_x.shape == (0, 3)



def test_count_fruitless():
    assert count_fruitless([12, 2, 3, 4]) == 2
    assert count_fruitless([0]) == 0
    assert count_fruitless([0, 12]) == 1
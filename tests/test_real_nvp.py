# %%
from rnpe.real_nvp import CouplingLayer, RealNVP
import jax.numpy as jnp
from jax import random
import pytest
import optax

def test_CouplingLayer():
    d=2
    D=5
    ac = CouplingLayer(
        d = d,
        D = D,
        hidden_features = [10, 10, 10, 10])

    key1, key2 = random.split(random.PRNGKey(0), 2)
    x = random.uniform(key1, (D,))

    params = ac.init(key2, x)
    y = ac.apply(params, x)[0]
    x_reconstructed = ac.apply(params, y, method=ac.inverse)

    assert pytest.approx(x, x_reconstructed)
    assert pytest.approx(x[:d], y[:d])
    assert (x[d:] != y[d:]).all()

# %%
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing
import jax.numpy as np

# %% Toy data
n_samples = 1000
plot_range = [(-2, 2), (-2, 2)]
n_bins = 100

scaler = preprocessing.StandardScaler()
X, _ = datasets.make_moons(n_samples=n_samples, noise=.05)
X = scaler.fit_transform(X)
plt.scatter(X[:, 0], X[:, 1], s=0.1)


# %%
from rnpe.real_nvp import CouplingLayer, RealNVP
import jax.numpy as jnp
from jax import random
import pytest

key = random.PRNGKey(0)
rnvp = RealNVP([5, 5], 5, 2)



# %%

def loss(params, x):
    z, log_abs_det = rnvp.apply(params, x)
    p_z = jax.scipy.stats.norm.pdf(z).sum()
    return -(p_z + log_abs_det)

# %%


# %%
import optax
import jax
params = rnvp.init(key, jnp.ones((2,)))
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

batch_loss1 = jax.jit(jax.vmap(loss, in_axes=[None, 0]))
batch_loss2 = lambda params, X : batch_loss1(params, X).sum()
batch_loss2(params, X)


# %%

def fit(params: optax.Params, optimizer: optax.GradientTransformation, batch):
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, batch):
        loss_value, grads = jax.value_and_grad(batch_loss2)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for epoch in range(10):
        for i, batch_idx in enumerate(range(0, X.shape[0], 50)):
            batch = X[batch_idx:(batch_idx+50)]
            params, opt_state, loss_value = step(params, opt_state, batch)
            if i % 20 == 0:
                print(f'step {i}, loss: {loss_value}')

    return params
  

optimizer = optax.adam(learning_rate=1e-2)
params = fit(params, optimizer, X)



# %%
from functools import partial

z = random.normal(key, (2, ))
rnvp.apply(params, z, method=rnvp.inverse)


# %%
import equinox as eqx
import functools as ft
import jax






# Start off by creating a model just like normal, but with some arbitrary Python
# objects as part of its parameterisation. In this case, we have `jax.nn.relu`, which
# is a Python function.

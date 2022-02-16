# %%
from torch.distributions import HalfCauchy, Normal, Independent
from torch import ones, zeros, manual_seed
manual_seed(1)

def sample_horseshoe(n: int = 2000, tau: float = 0.01, d: int = 2):
    lambdas = HalfCauchy(ones(d)).sample((n, ))
    eps_dist = Independent(Normal(zeros(d), tau*lambdas), 1)
    return eps_dist.sample()

# %%
eps = sample_horseshoe()
# %%
import matplotlib.pyplot as plt
plt.scatter(eps[:, 0], eps[:, 1])
plt.xlim(-0.5,0.5)
plt.ylim(-0.5,0.5)
# %%

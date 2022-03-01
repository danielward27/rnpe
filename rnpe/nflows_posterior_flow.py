
# Higher level abstractions for easier inference
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from rnpe.utils.train_flow import train_flow
from rnpe.nflows_nsf import get_coupling_nsf, get_autoregressive_nsf
from rnpe.utils.data import train_val_split_dataset
from nflows.flows import Flow


def train_posterior_flow(
    theta: Tensor,
    x: Tensor,
    max_patience: int = 3,
    lr: float = 5e-4,
    max_epochs: int = 20,
    batch_size = 128,
    val_prop = 0.1,
    flow_type = "coupling",
    **kwargs):
    """Train posterior neural spline flow q(theta|x).

    Args:
        theta (Tensor): Simulator parameters.
        x (Tensor): Simulations.
        max_patience (int, optional): Fruitless epochs before stopping training. Defaults to 3.
        lr (float, optional): Learning rate. Defaults to 5e-4.
        max_epochs (int, optional): Maximum epochs. Defaults to 20.
        batch_size (int, optional): Batch size. Defaults to 128.
        val_prop (float, optional): Validation set size. Defaults to 0.1.
        flow_type (str, optional): Either "autoregressive" or "coupling". Defaults to "autoregressive".
        kwargs: Key word arguments passed to `get_autoregressive_nsf` 

    Returns:
        tuple: (flow, training_logs)
    """
    train_data, val_data = train_val_split_dataset(TensorDataset(theta, x), val_prop=val_prop)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)

    flow_getter = {
        "autoregressive": get_autoregressive_nsf,
        "coupling": get_coupling_nsf
    }[flow_type.lower()]

    flow = flow_getter(
        theta.size(1),
        scale_means=theta.mean(axis=0),
        scale_stds=theta.std(axis=0),
        scale_means_cond=x.mean(axis=0),
        scale_stds_cond=x.std(axis=0),
        dim_cond=x.size(1),
        **kwargs
    )

    optimizer = Adam(flow.parameters(), lr=lr)

    flow, log = train_flow(
        flow,
        train_loader,
        val_loader,
        optimizer,
        max_epochs=max_epochs,
        max_patience=max_patience
    )
    return flow, log

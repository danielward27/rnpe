from typing import Optional
import torch
from torch import relu
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import (
    CompositeTransform,
    PiecewiseRationalQuadraticCouplingTransform,
    PiecewiseQuadraticCouplingTransform,
    LULinear,
    PointwiseAffineTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    MaskedPiecewiseQuadraticAutoregressiveTransform
)

from nflows.nn.nets import ResidualNet
from nflows.utils.torchutils import create_random_binary_mask
from rnpe.standardize import Standardize

def get_coupling_nsf(
    dim: int,
    scale_means: torch.Tensor,
    scale_stds: torch.Tensor,
    dim_cond: Optional[int] = None,
    scale_means_cond: Optional[torch.Tensor] = None,
    scale_stds_cond: Optional[torch.Tensor] = None,
    layers: int = 10,
    num_bins: int = 8,
    hidden_features: int = 50,
    tail_bound: float = 3.,
    tails="linear",
    spline: str = "rational_quadratic"
) -> torch.nn.Module:
    """Coupling neural spline flow."""
    if spline=="rational_quadratic":
        spline=PiecewiseRationalQuadraticCouplingTransform
    elif spline=="quadratic":
        spline=PiecewiseQuadraticCouplingTransform
    else:
        raise ValueError("Spline should be rational_quadratic or quadratic")

    base_dist = StandardNormal(shape=[dim])

    standardizing_transform = PointwiseAffineTransform(
        shift=-scale_means / scale_stds, scale=1 / scale_stds
    )

    conditioner = lambda in_features, out_features: ResidualNet(
        in_features=in_features,
        out_features=out_features,
        hidden_features=hidden_features,
        context_features=dim_cond,
        num_blocks=2,
        activation=relu,
    )

    transforms = [standardizing_transform]
    for _ in range(layers):
        layer = [
            spline(
                mask=create_random_binary_mask(features=dim),
                transform_net_create_fn=conditioner,
                num_bins=num_bins,
                tails=tails,
                tail_bound=tail_bound,
            ),
            LULinear(dim, identity_init=True),
        ]
        transforms.extend(layer)
    transforms = CompositeTransform(transforms)

    if dim_cond:
        if (scale_means_cond is None) or (scale_stds_cond is None):
            raise ValueError("scale_mean_cond and scale_stds_cond should be \
                provided to standardize the conditioning variable.")
        cond_standardizing_transform = Standardize(
            scale_means_cond, scale_stds_cond
        )
    else:
        cond_standardizing_transform = None
    
    flow = Flow(transforms, base_dist, cond_standardizing_transform)
    return flow



def get_autoregressive_nsf(
    dim: int,
    scale_means: torch.Tensor,
    scale_stds: torch.Tensor,
    dim_cond: Optional[int] = None,
    scale_means_cond: Optional[torch.Tensor] = None,
    scale_stds_cond: Optional[torch.Tensor] = None,
    layers: int = 10,
    num_bins: int = 8,
    hidden_features: int = 50,
    tail_bound: int = 3,
    tails="linear",
    spline: str = "rational_quadratic"
) -> torch.nn.Module:
    """Autoregressive neural spline flow."""
    if spline=="rational_quadratic":
        spline=MaskedPiecewiseRationalQuadraticAutoregressiveTransform
    elif spline=="quadratic":
        spline=MaskedPiecewiseQuadraticAutoregressiveTransform
    else:
        raise ValueError("Spline should be rational_quadratic or quadratic")

    base_dist = StandardNormal(shape=[dim])

    standardizing_transform = PointwiseAffineTransform(
        shift=-scale_means / scale_stds, scale=1 / scale_stds
    )

    transforms = [standardizing_transform]

    for _ in range(layers):
        layer = [
            spline(
                features=dim,
                hidden_features=hidden_features,
                context_features=dim_cond,
                num_bins=num_bins,
                tails=tails,
                tail_bound=tail_bound
            ),
            LULinear(dim, identity_init=True),
        ]
        transforms.extend(layer)
    transforms = CompositeTransform(transforms)

    if dim_cond:
        if (scale_means_cond is None) or (scale_stds_cond is None):
            raise ValueError("scale_mean_cond and scale_stds_cond should be \
                provided to standardize the conditioning variable.")
        cond_standardizing_transform = Standardize(
            scale_means_cond, scale_stds_cond
        )
    else:
        cond_standardizing_transform = None
    
    flow = Flow(transforms, base_dist, cond_standardizing_transform)
    return flow

"""Epps-Pulley univariate normality test against the standard normal."""

import torch
from torch import Tensor


def epps_pulley_statistic(samples: Tensor, t_max: float = 5.0, num_points: int = 256) -> Tensor:
    """Epps-Pulley statistic comparing each column's ECF to the standard normal CF.

    Accepts ``(N,)`` or ``(N, K)`` samples and returns ``()`` or ``(K,)``. The
    integrand is even in t, so we integrate over ``[0, t_max]`` and double rather than
    over the whole line. Samples are compared to N(0, 1) directly, without
    standardization, so location and scale deviations are detected.
    """
    squeeze = samples.dim() == 1
    if squeeze:
        samples = samples.unsqueeze(1)

    t = torch.linspace(0.0, t_max, num_points, device=samples.device, dtype=samples.dtype)
    arg = t[:, None, None] * samples[None, :, :]  # (T, N, K)
    cos_mean = torch.cos(arg).mean(dim=1)  # (T, K)
    sin_mean = torch.sin(arg).mean(dim=1)  # (T, K)
    normal_cf = torch.exp(-0.5 * t * t)[:, None]  # (T, 1)
    integrand = (cos_mean - normal_cf) ** 2 + sin_mean**2  # (T, K)
    stat = 2.0 * torch.trapezoid(integrand, t, dim=0)  # (K,)

    return stat[0] if squeeze else stat

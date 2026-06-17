"""Paired statistics for placement comparisons across shared seeds."""

import numpy as np
from scipy import stats


def paired_difference(
    baseline: list[float], treatment: list[float], n_boot: int = 10000, seed: int = 0
) -> dict[str, float]:
    """Paired comparison of treatment minus baseline across shared seeds.

    Both lists must be aligned by seed. Returns the mean difference, a paired t-test p-value,
    a 95 percent bootstrap confidence interval on the mean difference, and the sample size.
    A negative mean with a CI excluding zero means treatment is reliably lower (better for
    error metrics).
    """
    base = np.asarray(baseline, dtype=float)
    treat = np.asarray(treatment, dtype=float)
    diff = treat - base
    if diff.size < 2:
        raise ValueError("need at least two paired observations")

    _, p_value = stats.ttest_rel(treat, base)
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(diff, size=(n_boot, diff.size), replace=True).mean(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean_diff": float(diff.mean()),
        "p_value": float(p_value),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "n": int(diff.size),
    }

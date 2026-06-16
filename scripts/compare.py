"""Reproduce the placement comparison table (collapse and forecasting metrics).

Synthetic data by default:

    uv run python scripts/compare.py

On PEMS (download the .npz yourself):

    uv run python scripts/compare.py --pems /path/to/pems.npz
"""

import argparse

import numpy as np

from chronojepa.data import load_pems
from chronojepa.eval import (
    format_comparison_table,
    format_multiseed_table,
    run_multiseed_comparison,
    run_placement_comparison,
)


def _synthetic_series() -> np.ndarray:
    """A fixed multivariate synthetic series (seed 0) so the table is reproducible."""
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 8.0, 2400)[:, None]
    columns = [
        np.sin(2.0 * np.pi * 3.0 * t) + 0.3 * rng.standard_normal((2400, 1)),
        np.sin(2.0 * np.pi * 7.0 * t + 1.0) + 0.3 * rng.standard_normal((2400, 1)),
        np.cos(2.0 * np.pi * 2.0 * t) + 0.3 * rng.standard_normal((2400, 1)),
    ]
    return np.concatenate(columns, axis=1).astype(np.float32)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run the SIGReg placement comparison")
    parser.add_argument("--pems", help="path to a PEMS .npz; uses synthetic data if omitted")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-slices", type=int, default=48)
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument(
        "--forecast",
        choices=("mean", "trajectory"),
        default="mean",
        help="mean probes pooled features vs the horizon mean; trajectory probes the "
        "flattened token sequence vs the full horizon (temporally sensitive)",
    )
    parser.add_argument("--seeds", type=int, default=1, help="aggregate over seeds 0..N-1")
    parser.add_argument("--out", default="results/placement_comparison.json")
    args = parser.parse_args(argv)

    series = load_pems(args.pems) if args.pems else _synthetic_series()
    common = dict(
        placements=("pooled", "dual"),
        steps=args.steps,
        window=args.window,
        horizon=12,
        stride=8,
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_slices=args.num_slices,
        lam=0.5,
        forecast_mode=args.forecast,
    )

    if args.seeds > 1:
        results = run_multiseed_comparison(
            series, seeds=tuple(range(args.seeds)), results_path=args.out, **common
        )
        print(format_multiseed_table(results))
    else:
        results = run_placement_comparison(series, seed=0, results_path=args.out, **common)
        print(format_comparison_table(results))
    print(f"\nsaved: {args.out}")
    return results


if __name__ == "__main__":
    main()

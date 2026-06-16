"""Reproduce the placement comparison table (collapse and forecasting metrics).

Synthetic data by default:

    uv run python scripts/compare.py

On PEMS (download the .npz yourself):

    uv run python scripts/compare.py --pems /path/to/pems.npz
"""

import argparse

import numpy as np

from chronojepa.data import load_pems
from chronojepa.eval import format_comparison_table, run_placement_comparison


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
    parser.add_argument("--out", default="results/placement_comparison.json")
    args = parser.parse_args(argv)

    series = load_pems(args.pems) if args.pems else _synthetic_series()
    results = run_placement_comparison(
        series,
        placements=("pooled", "dual"),
        steps=args.steps,
        window=96,
        horizon=12,
        stride=8,
        batch_size=64,
        d_model=64,
        num_slices=48,
        lam=0.5,
        seed=0,
        results_path=args.out,
    )
    print(format_comparison_table(results))
    print(f"\nsaved: {args.out}")
    return results


if __name__ == "__main__":
    main()
